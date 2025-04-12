from ultralytics import YOLO
import os


# This script uses the YOLOv8 model to detect objects in an image and save the results.
# It requires the ultralytics package, which can be installed via pip.

IMG_PATH = "C:/Users/najms/car-scene-ai-assistant/images/scene1.jpg"
OUTPUT_DIR = 'C:/Users/najms/car-scene-ai-assistant/detections/'

model = YOLO("yolov8n.pt")

results = model(IMG_PATH)

results[0].show()

output_path = os.path.join(OUTPUT_DIR, 'scene1_detected.jpg')
results[0].save(filename=output_path)

print("Detected objects\n")
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    conf = float(box.conf[0])
    print(f"{label}: (conf:{conf:.2f})")

# Next step is to take the detected objects and summarize them structurally into a python dictionary.

from collections import defaultdict

object_counts = defaultdict(int)

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    object_counts[label] += 1

object_summary = dict(object_counts)

print("\nStructured scene summary:")
print(object_summary)



