from ultralytics import YOLO

yolo = YOLO("runs/detect/train/weights/last.pt",task='detect')

results = yolo(source="datasets/bgBlock_1611384368.png")