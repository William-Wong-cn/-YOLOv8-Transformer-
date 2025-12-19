# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO

path = 'models/best.pt'

video_path = "TestFiles/1.mp4"


model = YOLO(path)
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    success, frame = cap.read()

    if success:

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()