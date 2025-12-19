# -*- coding: utf-8 -*-
import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from ActionNet import YOLOTransformerAction
import Config
import random


BATCH_SIZE = 8
EPOCHS = 50
SEQ_LEN = 8
LR = 0.001
IMG_SIZE = 64
DATASET_ROOT = 'dataset/act-dataset'
MODEL_SAVE_PATH = 'action_transformer.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class ActionSequenceDataset(Dataset):
    """
    自定义数据集加载器：
    1. 读取 YOLO 格式的数据集
    2. 根据 txt 标签裁剪出人体图片
    3. 将图片打包成序列 (Sequence)
    """

    def __init__(self, root_dir, split='train', seq_len=8, transform=None):
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []


        img_dir = os.path.join(root_dir, split, 'images')
        label_dir = os.path.join(root_dir, split, 'labels')

        print(f"正在加载 {split} 数据集，请稍候...")
        print(f"图片路径: {img_dir}")


        img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                           glob.glob(os.path.join(img_dir, '*.png')))


        all_crops = []
        all_labels = []

        for img_path in img_files:

            basename = os.path.basename(img_path)
            name_only = os.path.splitext(basename)[0]
            label_path = os.path.join(label_dir, name_only + '.txt')

            if not os.path.exists(label_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape

            with open(label_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    data = lines[0].strip().split()
                    cls_id = int(data[0])
                    x_center = float(data[1]) * w
                    y_center = float(data[2]) * h
                    bbox_w = float(data[3]) * w
                    bbox_h = float(data[4]) * h

                    x1 = int(x_center - bbox_w / 2)
                    y1 = int(y_center - bbox_h / 2)
                    x2 = int(x_center + bbox_w / 2)
                    y2 = int(y_center + bbox_h / 2)

                    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    if crop.size == 0:
                        continue

                    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                    all_crops.append(crop)
                    all_labels.append(cls_id)

        for i in range(0, len(all_crops) - seq_len + 1, seq_len // 2):  # 重叠采样
            seq_crops = all_crops[i:i + seq_len]
            seq_labels = all_labels[i:i + seq_len]

            if len(set(seq_labels)) == 1:
                seq_tensor = torch.stack([self.transform(crop) for crop in seq_crops])
                self.samples.append((seq_tensor, seq_labels[0]))

        print(f"[-] 加载完成！总序列数: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return seq, label


def train():
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ActionSequenceDataset(root_dir=DATASET_ROOT, split='train', seq_len=SEQ_LEN, transform=transform)

    if len(train_dataset) == 0:
        print("错误：没有加载到数据。请检查 dataset/act-dataset/train/images 下是否有图片以及对应的 labels。")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    num_classes = len(Config.CH_names)
    print(f"初始化模型，类别数: {num_classes}, 序列长度: {SEQ_LEN}")

    model = YOLOTransformerAction(num_classes=num_classes, seq_len=SEQ_LEN)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("开始训练 Transformer 模型...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # inputs shape: [Batch, Seq_Len, 3, 64, 64]
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_acc = 100 * correct / total
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] 完成. 平均 Loss: {running_loss / len(train_loader):.4f}, 准确率: {epoch_acc:.2f}%")


    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"训练完成！模型已保存为: {MODEL_SAVE_PATH}")
    print("现在可以在 backend_detector.py 中使用这个模型权重了。")


if __name__ == '__main__':
    # 确保数据集路径存在
    if not os.path.exists(os.path.join(DATASET_ROOT, 'train')):
        print(f"错误：找不到数据集路径 {os.path.join(DATASET_ROOT, 'train')}")
        print("请确保项目结构如下：")
        print("dataset/")
        print("  act-dataset/")
        print("    train/")
        print("      images/ (存放图片)")
        print("      labels/ (存放txt)")
    else:
        train()