from ultralytics import YOLO
#
# yolo = YOLO("last.pt",task='detect')
#
# results = yolo(source="153042.png",save=True)

import cv2

model = YOLO("last.pt")
im2 = cv2.imread("153042.png")
results = model.predict(source=im2, show=True)