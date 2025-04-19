from ultralytics import YOLO
from collections import defaultdict
import os

model = YOLO("yolov8n.pt")

def detect_image(image_path, output_dir = "detections"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    results = model(image_path)

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, os.path.basename(image_path))
    results[0].save(filename = output_file_path)

    object_counts = defaultdict(int)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        object_counts[label] += 1

    return dict(object_counts), output_file_path
