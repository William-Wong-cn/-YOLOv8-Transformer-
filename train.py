# -*- coding: utf-8 -*-
from ultralytics import YOLO

# 加载预训练模型Load the pre-trained model
model = YOLO("yolov8n.pt")
# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data=r'D:\Workspace\python\PythonProject1\Heima\project\4\dataset\data.yaml', epochs=250, batch=8)