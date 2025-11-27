import importlib
import sys
import cv2
import numpy as np
from pathlib import Path

VIDEO_IN = "preview_uploaded.mp4"
OUT_VIDEO = "annotated_heatmap_output.mp4"
MAX_FRAMES = 500
MODEL_PATH = "yolov8n.pt"

# helper heatmap function
def generate_heatmap_from_boxes(frame_shape, boxes, radius=40, sigma=25):
    h, w = frame_shape[:2]
    heat = np.zeros((h, w), dtype=np.uint8)

    for b in boxes:
        try:
            x1, y1, x2, y2 = map(int, b[:4])
        except Exception:
            continue
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(heat, (cx, cy), radius, 255, -1)

    if heat.max() == 0:
        heat_blur = heat
    else:
        heat_blur = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma)
        heat_blur = np.uint8(255 * (heat_blur.astype(np.float32) / heat_blur.max()))
    heat_color = cv2.applyColorMap(heat_blur, cv2.COLORMAP_JET)
    return heat_color

# load ultralytics
try:
    ultralytics = importlib.import_module("ultralytics")
    YOLO = getattr(ultralytics, "YOLO")
except Exception as e:
    print("ultralytics import failed:", e)
    sys.exit(2)

if not Path(VIDEO_IN).exists():
    print(f"Input video {VIDEO_IN} not found. Aborting.")
    sys.exit(3)

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    print("Failed to open input video")
    sys.exit(4)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# output: side-by-side annotated + heatmap
out_w = w * 2
out_h = h
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (out_w, out_h))

print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded. Processing...")

frame_idx = 0
while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, classes=0)
    r = results[0]
    try:
        annotated = r.plot()
    except Exception:
        annotated = frame.copy()

    try:
        xy = r.boxes.xyxy
        coords = xy.cpu().numpy() if hasattr(xy, 'cpu') else np.array(xy)
    except Exception:
        coords = []

    heat = generate_heatmap_from_boxes(annotated.shape, coords)
    if heat.shape[:2] != annotated.shape[:2]:
        heat = cv2.resize(heat, (annotated.shape[1], annotated.shape[0]))

    combined = np.hstack((annotated, heat))
    out.write(combined)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
out.release()
print(f"Done. Wrote {OUT_VIDEO} ({frame_idx} frames).")
