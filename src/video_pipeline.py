from ultralytics import YOLO
import cv2
import os

VIDEO_PATH = "videos/sample_video.mp4"
OUTPUT_PATH = "detections/sample_video_detected.mp4"

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose = False)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

cap.release()
out.release()

print(f"Save detected file to {OUTPUT_PATH}")

