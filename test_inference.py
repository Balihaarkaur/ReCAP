import importlib
import sys
import cv2
import numpy as np
from pathlib import Path

VIDEO_CANDIDATES = ["preview_uploaded.mp4", "uploaded_video.mp4"]
OUT_ANNOT = "test_annotated.png"
OUT_HEAT = "test_heatmap.png"
MODEL_PATH = "yolov8n.pt"

# Load ultralytics
try:
    ultralytics = importlib.import_module("ultralytics")
    YOLO = getattr(ultralytics, "YOLO")
except Exception as e:
    print("ultralytics import failed:", e)
    sys.exit(2)

# Find a video file
video_file = None
for v in VIDEO_CANDIDATES:
    if Path(v).exists():
        video_file = v
        break

if video_file is None:
    print("No candidate video file found in workspace. Exiting.")
    sys.exit(3)

cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()
cap.release()
if not ret or frame is None:
    print("Failed to read first frame from", video_file)
    sys.exit(4)

print("Loaded frame from", video_file)

# Load model
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Failed to load model:", e)
    sys.exit(5)

print("Model loaded.")

# Run inference
try:
    results = model(frame, verbose=False, classes=0)
    r = results[0]
except Exception as e:
    print("Model inference failed:", e)
    sys.exit(6)

# Annotated
try:
    annotated = r.plot()
except Exception:
    annotated = frame.copy()

# Boxes -> heatmap
try:
    xy = r.boxes.xyxy
    coords = xy.cpu().numpy() if hasattr(xy, 'cpu') else np.array(xy)
except Exception:
    coords = []

h, w = annotated.shape[:2]
heat = np.zeros((h, w), dtype=np.uint8)
for b in coords:
    try:
        x1, y1, x2, y2 = map(int, b[:4])
    except Exception:
        continue
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    cv2.circle(heat, (cx, cy), 40, 255, -1)

if heat.max() == 0:
    heat_blur = heat
else:
    heat_blur = cv2.GaussianBlur(heat, (0, 0), sigmaX=25)
    heat_blur = np.uint8(255 * (heat_blur.astype(np.float32) / heat_blur.max()))
heat_color = cv2.applyColorMap(heat_blur, cv2.COLORMAP_JET)

# Save outputs
cv2.imwrite(OUT_ANNOT, annotated)
cv2.imwrite(OUT_HEAT, heat_color)
print("Wrote:", OUT_ANNOT, OUT_HEAT)
print("Person count:", len(r.boxes))
