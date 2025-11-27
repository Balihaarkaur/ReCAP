import importlib
import sys
import time
import cv2
import numpy as np

# load ultralytics YOLO from venv
try:
    ultralytics = importlib.import_module("ultralytics")
    YOLO = getattr(ultralytics, "YOLO")
except Exception as e:
    print("ultralytics not available:", e)
    sys.exit(1)

# helper from app.py (copied)

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


def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam (index 0)")
        sys.exit(1)

    print("Press 'q' to quit the headless runner.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break

        results = model(frame, verbose=False, classes=0)
        result = results[0]
        # annotated frame
        try:
            annotated = result.plot()
        except Exception:
            annotated = frame.copy()

        # get boxes
        try:
            xy = result.boxes.xyxy
            coords = xy.cpu().numpy() if hasattr(xy, "cpu") else np.array(xy)
        except Exception:
            coords = []

        heat_color = generate_heatmap_from_boxes(annotated.shape, coords)

        # resize heatmap to match annotated if necessary
        if heat_color.shape[:2] != annotated.shape[:2]:
            heat_color = cv2.resize(heat_color, (annotated.shape[1], annotated.shape[0]))

        # show side-by-side
        combined = np.hstack((annotated, heat_color))
        cv2.imshow('Annotated | Heatmap', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
