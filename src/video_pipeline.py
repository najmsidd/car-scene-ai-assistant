from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

def detect_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

    print(f"Save detected file to {output_path}")

