from ultralytics import YOLO
from collections import defaultdict
import os

model = YOLO("yolov8n.pt")

def detect_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    results = model(image_path)

    object_counts = defaultdict(int)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        object_counts[label] += 1

    return dict(object_counts)
