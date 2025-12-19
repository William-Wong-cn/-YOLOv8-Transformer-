# -*- coding: utf-8 -*-
import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import time
import torchvision.transforms as T
from collections import deque, defaultdict
from ActionNet import YOLOTransformerAction
import Config
import matplotlib.pyplot as plt
import detect_tools as tools

# 配置
VIDEO_FOLDER = 'TestFiles'
MODEL_PATH = Config.model_path
TRANSFORMER_WEIGHT = 'action_transformer.pth'
SEQ_LEN = 8
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = Config.CH_names

class VideoDetector:
    def __init__(self):
        print("正在加载 YOLO 模型...")
        self.model_yolo = YOLO(MODEL_PATH)

        print("正在加载 Transformer 动作识别模型...")
        self.transformer = YOLOTransformerAction(num_classes=len(CLASS_NAMES), seq_len=SEQ_LEN)
        self.transformer.to(DEVICE)
        if os.path.exists(TRANSFORMER_WEIGHT):
            self.transformer.load_state_dict(torch.load(TRANSFORMER_WEIGHT, map_location=DEVICE))
            print(f"Transformer 权重加载成功: {TRANSFORMER_WEIGHT}")
        else:
            print(f"警告: 未找到 {TRANSFORMER_WEIGHT}，将使用随机权重")
        self.transformer.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sequence_buffer = deque(maxlen=SEQ_LEN)
        self.processed_videos = set()
        self.current_video_stats = defaultdict(list)  # {action: [conf1, conf2, ...]}

    def draw_action_label(self, frame, x1, y1, x2, y2, action_name, conf):

        label = f"{action_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.rectangle(frame, (x1, y1 - 40), (x1 + len(label)*14 + 20, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 2)
        return frame

    def show_and_save_accuracy_chart(self, video_name):

        if not self.current_video_stats:
            print("本视频无动作识别记录，跳过图表生成。")
            return

        actions = list(self.current_video_stats.keys())
        avg_confs = [np.mean(self.current_video_stats[a]) for a in actions]
        counts = [len(self.current_video_stats[a]) for a in actions]

        # 支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 7))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD']
        bars = plt.bar(actions, avg_confs, color=colors[:len(actions)], edgecolor='black', linewidth=1.2, alpha=0.9)
        plt.title(f'视频动作识别准确率分析\n{os.path.basename(video_name)}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('动作类别', fontsize=12)
        plt.ylabel('平均置信度（越高越准确）', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.4)

        for bar, avg, count in zip(bars, avg_confs, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{avg:.3f}\n({count}次)', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()


        os.makedirs(Config.save_path, exist_ok=True)
        save_path = os.path.join(Config.save_path, f"{os.path.splitext(os.path.basename(video_name))[0]}_accuracy.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
        print(f"准确率图表已保存: {save_path}")


        plt.show()

        plt.close()

    def detect_video(self, video_path):
        print(f"\n正在处理视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频！")
            return

        self.current_video_stats.clear()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model_yolo(frame, conf=0.25, verbose=False)[0]
            boxes = results.boxes.data.cpu().numpy() if results.boxes is not None else []

            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = map(float, box)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                crop_tensor = self.transform(crop).unsqueeze(0)
                self.sequence_buffer.append(crop_tensor)

                if len(self.sequence_buffer) == SEQ_LEN:
                    seq = torch.cat(list(self.sequence_buffer), dim=0).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        logits = self.transformer(seq)
                        probs = torch.softmax(logits, dim=1)
                        pred_id = torch.argmax(probs, dim=1).item()
                        pred_conf = probs[0, pred_id].item()

                    action_name = CLASS_NAMES[pred_id]
                    self.current_video_stats[action_name].append(pred_conf)


                    frame = self.draw_action_label(frame, x1, y1, x2, y2, action_name, pred_conf)

            cv2.imshow(f'动作识别 - {os.path.basename(video_path)}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


        print("\n=== 本视频动作识别结果 ===")
        for action, confs in self.current_video_stats.items():
            print(f"{action}: 平均置信度 {np.mean(confs):.3f} （共 {len(confs)} 次）")


        self.show_and_save_accuracy_chart(video_path)

    def monitor_folder(self):
        print(f"正在监控文件夹: {VIDEO_FOLDER}")
        print("请将视频放入 TestFiles 文件夹，程序会自动检测...")
        while True:
            for file in os.listdir(VIDEO_FOLDER):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                    video_path = os.path.join(VIDEO_FOLDER, file)
                    if video_path not in self.processed_videos:
                        self.detect_video(video_path)
                        self.processed_videos.add(video_path)
            time.sleep(3)

if __name__ == "__main__":
    detector = VideoDetector()
    detector.monitor_folder()