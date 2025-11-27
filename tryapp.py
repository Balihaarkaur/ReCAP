import streamlit as st
import cv2
import numpy as np
import importlib
import sys
import subprocess
from pathlib import Path
import time

# ========================================================
# Configuration
# ========================================================

MAX_CAPACITY = 500
SAFE_THRESHOLD = 0.65
DENSE_THRESHOLD = 0.95
PERSON_CLASS_ID = 0

# Backwards-compatible cache decorator for older Streamlit versions
if hasattr(st, "cache_resource"):
    cache_resource = st.cache_resource
elif hasattr(st, "experimental_singleton"):
    cache_resource = st.experimental_singleton
else:
    # Fallback no-op decorator
    def cache_resource(func):
        return func


# Load YOLO Model
@cache_resource
def load_model(path: str = "yolov8n.pt"):
    """Load and cache the YOLO model. This avoids blocking Streamlit on import and
    prevents re-loading the model on every rerun."""
    try:
        ultralytics = importlib.import_module("ultralytics")
        YOLO = getattr(ultralytics, "YOLO")
    except Exception:
        # Raise a controlled error so higher-level UI can show an install button
        raise ModuleNotFoundError("ultralytics")

    return YOLO(path)


# ========================================================
# Core Functions
# ========================================================

def calculate_status(count, safe_threshold: float = SAFE_THRESHOLD, dense_threshold: float = DENSE_THRESHOLD):
    ratio = count / MAX_CAPACITY
    if ratio >= dense_threshold:
        return "Overcrowded"
    elif ratio >= safe_threshold:
        return "Dense"
    else:
        return "Safe"


def apply_heatmap(frame, status):
    """Overlay heatmap color based on status"""
    if status == "Safe":
        color = (0, 255, 0)
        intensity = 0.25
    elif status == "Dense":
        color = (0, 165, 255)
        intensity = 0.45
    else:
        color = (0, 0, 255)
        intensity = 0.55

    overlay = np.full(frame.shape, color, dtype=np.uint8)
    return cv2.addWeighted(frame, 1.0, overlay, intensity, 0)


def generate_heatmap_from_boxes(frame_shape, boxes, radius=40, sigma=25):
    """Generate a colored heatmap image from detected bounding boxes.

    - frame_shape: shape of the frame (h, w, channels)
    - boxes: iterable of [x1,y1,x2,y2,...]
    """
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


# ========================================================
# Streamlit Interface
# ========================================================

st.title("🎥 Crowd Monitoring System with YOLOv8 + Heatmap")
st.write("Upload a video or use webcam to detect crowd level with heatmap overlay.")

source = st.radio("Select Input Source", ["Upload Video", "Webcam"])

uploaded_video = None
if source == "Upload Video":
    uploaded_video = st.file_uploader("Upload your crowd video", type=["mp4", "avi", "mov"])

    # If a video is uploaded, offer a one-shot preview (first frame -> annotated + heatmap)
    if uploaded_video is not None:
        st.markdown("**Preview first frame (generate heatmap)**")
        if st.button("Generate Preview Heatmap"):
            tfile_preview = "preview_uploaded.mp4"
            with open(tfile_preview, "wb") as f:
                f.write(uploaded_video.read())

            cap_preview = cv2.VideoCapture(tfile_preview)
            ret, frame_preview = cap_preview.read()
            cap_preview.release()

            if not ret or frame_preview is None:
                st.error("Could not read first frame from uploaded video.")
            else:
                with st.spinner("Running model on preview frame..."):
                    try:
                        model_preview = load_model()
                    except ModuleNotFoundError:
                        st.error("The 'ultralytics' package is not installed in this environment.")
                    else:
                        results_preview = model_preview(frame_preview, verbose=False, classes=PERSON_CLASS_ID)
                        r = results_preview[0]
                        try:
                            annotated_prev = r.plot()
                        except Exception:
                            annotated_prev = frame_preview.copy()

                        try:
                            xy_prev = r.boxes.xyxy
                            coords_prev = xy_prev.cpu().numpy() if hasattr(xy_prev, "cpu") else np.array(xy_prev)
                        except Exception:
                            coords_prev = []

                        heat_prev = generate_heatmap_from_boxes(annotated_prev.shape, coords_prev)
                        annotated_rgb_prev = cv2.cvtColor(annotated_prev, cv2.COLOR_BGR2RGB)
                        heat_rgb_prev = cv2.cvtColor(heat_prev, cv2.COLOR_BGR2RGB)

                        c1, c2 = st.columns([2,1])
                        c1.image(annotated_rgb_prev, channels="RGB", caption="Annotated Preview")
                        c2.image(heat_rgb_prev, channels="RGB", caption="Preview Heatmap")

run_button = st.button("Start Analysis")
stop_button = st.button("Stop Analysis")

# session state for run/stop control
if "running" not in st.session_state:
    st.session_state.running = False

if run_button:
    # ensure ultralytics is available and if not, offer an install button
    try:
        importlib.import_module("ultralytics")
        ultralytics_missing = False
    except Exception:
        ultralytics_missing = True

    if ultralytics_missing:
        st.error("The 'ultralytics' package is not available in the active environment.")
        install = st.button("Install ultralytics now (may take a few minutes)")
        if install:
            st.info("Installing ultralytics into the virtualenv. This may take several minutes.")
            # try to use the current venv python executable
            venv_python = sys.executable
            try:
                subprocess.check_call([venv_python, "-m", "pip", "install", "ultralytics>=8.0.0"])
                st.success("Installed ultralytics. Attempting to import now...")
                try:
                    importlib.import_module("ultralytics")
                    ultralytics_missing = False
                except Exception:
                    st.error("Installed ultralytics but import still fails. Please restart the app to finish installation.")
                    st.stop()
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to install ultralytics: {e}")
                st.stop()

    st.session_state.running = True
if stop_button:
    st.session_state.running = False


# ========================================================
# Video Processing Logic
# ========================================================

if run_button:
    stframe = st.empty()  # Streamlit live frame container

    # Check ultralytics availability before attempting to load model
    try:
        importlib.import_module("ultralytics")
        ultralytics_missing = False
    except Exception:
        ultralytics_missing = True

    if ultralytics_missing:
        st.error("The 'ultralytics' package is not available in the active environment.")
        install = st.button("Install ultralytics now (may take a few minutes)")
        if install:
            st.info("Installing ultralytics into the virtualenv. This may take several minutes.")
            venv_python = sys.executable
            try:
                subprocess.check_call([venv_python, "-m", "pip", "install", "ultralytics>=8.0.0"])
                st.success("Installed ultralytics. Attempting to import now...")
                try:
                    importlib.import_module("ultralytics")
                    ultralytics_missing = False
                except Exception:
                    st.error("Installed ultralytics but import still fails. Please restart the app to finish installation.")
                    st.stop()
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to install ultralytics: {e}")
                st.stop()
        else:
            st.stop()

    # Load (or get cached) model with a spinner so UI remains responsive
    with st.spinner("Loading YOLO model (may download weights)..."):
        model = load_model()

    # Select video source
    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if uploaded_video is None:
            st.error("Please upload a video first.")
            st.session_state.running = False
            st.stop()

        tfile = "uploaded_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile)

    # Prepare side-by-side placeholders: left = video, right = heatmap
    col_video, col_heat = st.columns([2, 1])
    vid_placeholder = col_video.empty()
    heat_placeholder = col_heat.empty()

    # Run processing loop while the cap is open and user hasn't stopped
    while cap.isOpened() and st.session_state.running:

        ret, frame = cap.read()
        if not ret:
            st.warning("Video ended or stream unavailable.")
            break

        # Inference
        results = model(frame, verbose=False, classes=PERSON_CLASS_ID)
        result = results[0]
        person_count = len(result.boxes)

        # Generate bounding box frame
        annotated = result.plot()

        # Determine crowd status
        status = calculate_status(person_count)

        # Apply heatmap overlay
        annotated = apply_heatmap(annotated, status)

        # Add text
        text = f"Count: {person_count}/{MAX_CAPACITY} | Status: {status}"
        color = (0,255,0) if status=="Safe" else (0,165,255) if status=="Dense" else (0,0,255)
        cv2.putText(annotated, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Generate a per-frame heatmap from detected boxes
        try:
            xy = result.boxes.xyxy
            coords = xy.cpu().numpy() if hasattr(xy, "cpu") else np.array(xy)
        except Exception:
            coords = []

        heat_color = generate_heatmap_from_boxes(annotated.shape, coords)

        # Convert BGR → RGB for Streamlit
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        # Update placeholders side-by-side
        vid_placeholder.image(annotated_rgb, channels="RGB")
        heat_placeholder.image(heat_rgb, channels="RGB")

        # allow Streamlit to handle other events (like Stop button)
        time.sleep(0.01)

    cap.release()
    st.session_state.running = False
    st.success("Analysis complete.")


    