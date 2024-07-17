

from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)

results = model.train(data='data', epochs=1, imgsz=64)
