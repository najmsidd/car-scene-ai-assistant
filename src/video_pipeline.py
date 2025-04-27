from ultralytics import YOLO
import cv2
import os
from collections import defaultdict
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

def detect_video(input_path, output_path, frame_interval = 30):
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video...")

    frame_count = 0
    final_counts = defaultdict(int)
    counted_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose = False)
        detections = results[0]

        det_list = []
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            det_list.append(([x1, y1, x2-x1, y2-y1], conf, cls))

        tracks = tracker.update_tracks(det_list, frame= frame)

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls = track.get_det_class()
            label = detections.names[int(cls)]

            if track_id not in counted_ids:
                final_counts[label] += 1
                counted_ids.add(track_id)

            l,t,w,h = track.to_ltrb()
            r, b = l+w, t+h
            cv2.rectangle(frame, (int(l), int(t), int(r), int(b)), (0,255,0), 2)
            cv2.putText(frame, f'{label} ID:{track_id}',(int(l), int(t)-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()

    print(f"Saved detected video to {output_path}")
    return dict(final_counts)

        


