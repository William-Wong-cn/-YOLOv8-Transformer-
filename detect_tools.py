# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os

def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def drawRectBox(image, rect, addText, color=(0, 255, 0), font_size=None):
    """
    绘制矩形框与结果
    """
    # 绘制位置方框
    cv2.rectangle(image, (rect[0], rect[1]),
                  (rect[2], rect[3]),
                  color, 2)

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    font_path = 'Font/platech.ttf' if os.path.exists('Font/platech.ttf') else None
    if font_path:
        if font_size is None:
            font_size = int((rect[2] - rect[0]) / 8)
        font = ImageFont.truetype(font_path, font_size)
        draw.text((rect[0], rect[3]), addText, (0, 255, 0), font=font)
    imagex = np.array(img)
    return imagex


def img_cvread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def draw_boxes(img, boxes):
    for each in boxes:
        x1 = each[0]
        y1 = each[1]
        x2 = each[2]
        y2 = each[3]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def yolo_to_location(w, h, yolo_data):
    x_, y_, w_, h_ = yolo_data
    x1 = int(w * x_ - 0.5 * w * w_)
    x2 = int(w * x_ + 0.5 * w * w_)
    y1 = int(h * y_ - 0.5 * h * h_)
    y2 = int(h * y_ + 0.5 * h * h_)
    return [x1, y1, x2, y2]

def location_to_yolo(w, h, locations):
    x1, y1, x2, y2 = locations
    x_ = (x1 + x2) / 2 / w
    x_ = float('%.5f' % x_)
    y_ = (y1 + y2) / 2 / h
    y_ = float('%.5f' % y_)
    w_ = (x2 - x1) / w
    w_ = float('%.5f' % w_)
    h_ = (y2 - y1) / h
    h_ = float('%.5f' % h_)
    return [x_, y_, w_, h_]