import os

import torch
from ultralytics import YOLO

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda_available = torch.cuda.is_available()
    print(cuda_available)

    model = YOLO('yolov8n.pt')

    results = model.train(data='huakuai_train.yaml',epochs=30,imgsz=312, device='0')
    results = model.val()
    success = model.export(format='onnx')
