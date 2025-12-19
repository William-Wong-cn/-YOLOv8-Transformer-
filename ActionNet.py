# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Transformer 需要位置编码来理解“第几帧”
    """

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        x = x + self.pe[:, :x.size(1), :]
        return x


class YOLOTransformerAction(nn.Module):
    def __init__(self, num_classes=6, d_model=128, nhead=4, num_layers=2, seq_len=10):
        super(YOLOTransformerAction, self).__init__()


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),  # [B, 128, 8, 8]
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 128, 1, 1] -> Flatten -> [B, 128]
            nn.Flatten()
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)


        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input x shape: [Batch, Seq_Len, 3, 64, 64] (图像序列)
        batch_size, seq_len, C, H, W = x.size()


        x = x.view(batch_size * seq_len, C, H, W)


        features = self.feature_extractor(x)  # [Batch*Seq_Len, d_model]


        features = features.view(batch_size, seq_len, -1)  # [Batch, Seq_Len, d_model]


        features = self.pos_encoder(features)


        transformer_out = self.transformer_encoder(features)  # [Batch, Seq_Len, d_model]


        final_feature = transformer_out[:, -1, :]


        logits = self.classifier(final_feature)

        return logits



if __name__ == "__main__":
    model = YOLOTransformerAction(num_classes=6, seq_len=8)
    # 模拟输入：Batch=1, 序列长度=8帧, 3通道, 64x64大小
    dummy_input = torch.randn(1, 8, 3, 64, 64)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 应该是 [1, 6]