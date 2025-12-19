# -*- coding: utf-8 -*-
#Camera detection and initialization,
# YOLOv8 object detection,
# real-time video display
import cv2
from ultralytics import YOLO


path = 'models/best.pt'


model = YOLO(path)

ID = 0

while(ID<10):
    cap = cv2.VideoCapture(ID)

    ret, frame = cap.read()
    if ret == False:
        ID += 1
    else:
        print('摄像头ID:',ID)
        break



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