import streamlit as st
import cv2
import numpy as np
import importlib
import sys
import subprocess
from pathlib import Path
import time
import logging
import streamlit.components.v1 as components
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

# ========================================================
# Configuration
# ========================================================

MAX_CAPACITY = 500
SAFE_THRESHOLD = 0.65
DENSE_THRESHOLD = 0.95
PERSON_CLASS_ID = 0

# Load YOLO Model
@st.cache_resource
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
    """Return status string based on thresholds provided.

    - `safe_threshold` and `dense_threshold` are ratios (0.0-1.0) of MAX_CAPACITY.
    """
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


def generate_activation_cam(model, frame, boxes=None, target_size=640):
    """Generate a detection-weighted CAM-like heatmap.

    Strategy:
    - Run a forward pass through the model and capture the last Conv2d feature map.
    - If `boxes` are provided (Nx4 array in image coordinates), map them to the
      feature map spatial dimensions and compute per-channel weights as the mean
      activation inside each box. Combine channels using those weights (score-CAM
      style but without extra forward passes) so activations inside detected
      person boxes contribute more strongly.
    - If no boxes are given or weights are all zero, fall back to channel-average CAM.
    - Upsample to original frame size, apply slight blur and a Jet colormap.
    """
    if torch is None or nn is None:
        raise RuntimeError("PyTorch unavailable for CAM generation")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0)

    # Find last Conv2d in model
    module_root = getattr(model, "model", model)
    target_layer = None
    for m in module_root.modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in model")

    activations = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach().cpu()

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        # Move input to model device
        device = next(module_root.parameters()).device if any(True for _ in module_root.parameters()) else torch.device('cpu')
        img_t = img_t.to(device)
        # Run forward through module_root (backbone head etc.)
        try:
            _ = module_root(img_t)
        except Exception:
            # Try calling the top-level model if module_root isn't callable
            _ = model(img_resized)

        handle.remove()

        if 'value' not in activations:
            raise RuntimeError("Activation hook did not capture any tensor")

        feat = activations['value'][0]  # C x Hf x Wf (on CPU)
        C, Hf, Wf = feat.shape

        # If boxes provided, compute channel weights from activations inside mapped boxes
        weights = None
        if boxes is not None and len(boxes) > 0:
            weights = torch.zeros(C, dtype=torch.float32)
            count_valid = 0
            for b in boxes:
                try:
                    x1, y1, x2, y2 = map(int, b[:4])
                except Exception:
                    continue
                # map box coords from original image (w,h) to feature map (Wf,Hf)
                fx1 = max(0, int(x1 * Wf / w))
                fy1 = max(0, int(y1 * Hf / h))
                fx2 = min(Wf, int(x2 * Wf / w) + 1)
                fy2 = min(Hf, int(y2 * Hf / h) + 1)
                if fx2 <= fx1 or fy2 <= fy1:
                    continue
                region = feat[:, fy1:fy2, fx1:fx2]
                if region.numel() == 0:
                    continue
                # mean activation per channel inside this box
                w_k = region.view(C, -1).mean(dim=1)
                weights += w_k
                count_valid += 1
            if count_valid > 0:
                weights = weights / float(count_valid)
            else:
                weights = None

        # Compute CAM: either weighted sum of channels or plain average
        if weights is not None and weights.sum() > 0:
            # convert to numpy for processing
            feat_np = feat.numpy()
            w_np = weights.numpy()
            cam_map = np.tensordot(w_np, feat_np, axes=(0, 0))  # Hf x Wf
        else:
            cam_map = feat.mean(dim=0).numpy()

        # Normalize
        cam_map -= cam_map.min()
        if cam_map.max() > 0:
            cam_map = cam_map / cam_map.max()
        cam_uint8 = np.uint8(255 * cam_map)
        cam_resized = cv2.resize(cam_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
        # Slight blur to smooth
        cam_resized = cv2.GaussianBlur(cam_resized, (11, 11), 0)
        cam_color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        return cam_color
    finally:
        try:
            handle.remove()
        except Exception:
            pass


# ========================================================
# Streamlit Interface
# ========================================================

# Use a wide page layout and left-align content so information is spread across the page
st.set_page_config(page_title="UrbanCrowd Insight", layout="wide")

# Small CSS adjustments to make the app use full width and left-aligned headings
st.markdown(
    """
    <style>
      .block-container{max-width:100% !important; padding-left:2rem; padding-right:2rem}
      h1, .stTitle {text-align: left !important}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='margin:0 0 0.5rem 0'>🎥 UrbanCrowd Insight – Heatmap Based Crowd Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='margin-top:0.25rem;margin-bottom:1rem;'>Upload a video or use webcam to detect crowd level with heatmap overlay.</p>", unsafe_allow_html=True)

# Build a native, functional control panel that mirrors the React UI
# Spread the UI across three evenly spaced columns so content is not centered
left_col, center_col, right_col = st.columns([1, 1, 1])

# assign named aliases used later in the code
control_col = left_col
preview_col = center_col

with control_col:
    st.subheader("CrowdSense Controls")

    # Try to use the React/Streamlit component if available. If the
    # component isn't available (no package or dev server), fall back to
    # the native Streamlit controls below so the app remains functional.
    uploaded_video = None
    try:
        from streamlit_crowdsense_component import crowdsense_component
        comp_default = {
            "running": st.session_state.get("running", False),
            "densityThreshold": st.session_state.get("density_thresh", int(SAFE_THRESHOLD * 100)),
            "sensitivity": st.session_state.get("sensitivity", 0.5),
            "source": "Upload Video",
        }
        comp_val = crowdsense_component(default=comp_default, key="crowdsense_comp")

        # If the component returns a value, sync it into session_state
        if comp_val:
            # source
            source = comp_val.get("source", "Upload Video")
            if source not in ("Upload Video", "Webcam"):
                source = "Upload Video"
            # uploaded video placeholder is still handled by native uploader
            st.session_state.running = bool(comp_val.get("running", st.session_state.get("running", False)))
            st.session_state.density_thresh = int(comp_val.get("densityThreshold", st.session_state.get("density_thresh", int(SAFE_THRESHOLD * 100))))
            st.session_state.sensitivity = float(comp_val.get("sensitivity", st.session_state.get("sensitivity", 0.5)))
        else:
            # No value returned by component yet (e.g. dev server not running)
            source = "Upload Video"
    except Exception:
        # Component not available; use native controls
        source = st.radio("Select Input Source", ["Upload Video", "Webcam"], index=0)

    if source == "Upload Video":
        uploaded_video = st.file_uploader("Upload your crowd video", type=["mp4", "avi", "mov"])

    # Ensure session defaults
    if "density_thresh" not in st.session_state:
        st.session_state.density_thresh = int(SAFE_THRESHOLD * 100)
    if "sensitivity" not in st.session_state:
        st.session_state.sensitivity = 0.50

    # Render native sliders so values are editable even if component isn't connected
    st.session_state.density_thresh = st.slider("Density Threshold (%)", 1, 100, st.session_state.density_thresh)
    st.session_state.sensitivity = st.slider("Sensitivity", 0.10, 1.00, float(st.session_state.sensitivity), step=0.05)

    st.markdown("---")
    if "running" not in st.session_state:
        st.session_state.running = False

    # Native start/stop buttons remain useful; component can also toggle running
    start_clicked = st.button("Start")
    stop_clicked = st.button("Stop")

    if start_clicked:
        st.session_state.running = True
    if stop_clicked:
        st.session_state.running = False

    st.markdown("---")
    st.markdown("**Preview first frame (generate heatmap)**")
    if source == "Upload Video" and uploaded_video is not None:
        if st.button("Generate Preview"):
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

                        try:
                            heat_prev = generate_activation_cam(model_preview, annotated_prev, boxes=coords_prev)
                        except Exception:
                            heat_prev = generate_heatmap_from_boxes(annotated_prev.shape, coords_prev)
                        annotated_rgb_prev = cv2.cvtColor(annotated_prev, cv2.COLOR_BGR2RGB)
                        heat_rgb_prev = cv2.cvtColor(heat_prev, cv2.COLOR_BGR2RGB)

                        c1, c2 = preview_col.columns([2,1])
                        c1.image(annotated_rgb_prev, channels="RGB", caption="Annotated Preview")
                        c2.image(heat_rgb_prev, channels="RGB", caption="Preview Heatmap")
    elif source == "Upload Video":
        st.info("Upload a video to enable preview generation.")

# Initialize history storage in session state
if "crowd_history" not in st.session_state:
        st.session_state.crowd_history = []
if "time_history" not in st.session_state:
        st.session_state.time_history = []

# Dashboard: use Streamlit native metrics + sparkline and download tools
with preview_col:
    # Metrics row (Current Count, Density, Status) with bold headings and larger values
    m1, m2, m3 = st.columns(3)

    # Headings: bold and slightly larger
    m1.markdown("<div style='font-weight:700; font-size:18px; text-align:left;'>Current Count</div>", unsafe_allow_html=True)
    m2.markdown("<div style='font-weight:700; font-size:18px; text-align:left;'>Density</div>", unsafe_allow_html=True)
    m3.markdown("<div style='font-weight:700; font-size:18px; text-align:left;'>Status</div>", unsafe_allow_html=True)

    # Placeholders for in-place updates (used by the processing loop)
    count_placeholder = m1.empty()
    density_placeholder = m2.empty()
    status_placeholder = m3.empty()

    # Initialize placeholders as empty large-value areas (will be updated during processing)
    count_placeholder.markdown("<div style='font-size:36px; font-weight:800; text-align:left;'>&nbsp;</div>", unsafe_allow_html=True)
    density_placeholder.markdown("<div style='font-size:28px; font-weight:700; text-align:left;'>&nbsp;</div>", unsafe_allow_html=True)
    status_placeholder.markdown("<div style='font-size:26px; font-weight:700; text-align:left;'>&nbsp;</div>", unsafe_allow_html=True)

    # Sparkline / history chart
    st.markdown("**Crowd History**")
    history_chart = st.empty()

    # Download buttons area
    st.markdown("**Downloads**")
    dl_col1, dl_col2 = st.columns(2)
    # placeholders for download buttons; actual content set below when data exists
    dl_latest_annot = dl_col1.empty()
    dl_latest_heat = dl_col2.empty()

    # CSV export for history
    csv_col = st.container()

# Start/Stop handled by the native control panel above (session_state.running)

# Configure basic logging to a file for diagnostics
log_file = Path(__file__).parent / "streamlit_error.log"
logging.basicConfig(level=logging.INFO, filename=str(log_file), filemode="a",
                    format="%(asctime)s [%(levelname)s] %(message)s")


# ========================================================
# Video Processing Logic
# ========================================================

def _process_stream():
    """Encapsulated processing flow so we can call it under session-state control
    and catch/log any exceptions without letting Streamlit kill the runtime."""

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
                    return
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to install ultralytics: {e}")
                raise
        else:
            return

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
            return

        tfile = "uploaded_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile)

    # Prepare side-by-side placeholders: left = video, right = heatmap
    col_video, col_heat = st.columns([2, 1])
    vid_placeholder = col_video.empty()
    heat_placeholder = col_heat.empty()

    # Run processing loop while the cap is open and user hasn't stopped
    try:
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

            # Determine crowd status using user-controlled density threshold
            density_threshold_percent = st.session_state.get("density_thresh", int(SAFE_THRESHOLD*100))
            user_safe_thresh = density_threshold_percent / 100.0
            # Keep dense threshold somewhat above safe threshold (cap at 0.99)
            user_dense_thresh = min(0.99, user_safe_thresh + 0.30)
            status = calculate_status(person_count, safe_threshold=user_safe_thresh, dense_threshold=user_dense_thresh)

            # Update dashboard stats and session history
            try:
                density_percent = int((person_count / MAX_CAPACITY) * 100)
            except Exception:
                density_percent = 0

            # Append history (keep it bounded)
            st.session_state.crowd_history.append(person_count)
            st.session_state.time_history.append(time.time())
            if len(st.session_state.crowd_history) > 2000:
                st.session_state.crowd_history.pop(0)
                st.session_state.time_history.pop(0)

            # Update the stat-card placeholders via small JS snippet so they update in-place
            # Update native placeholders (metrics, chart, downloads)
            try:
                # update metrics if placeholders exist (left-aligned, larger font)
                count_placeholder.markdown(f"<div style='font-size:40px;font-weight:800;text-align:left;'>{person_count}</div>", unsafe_allow_html=True)
                density_placeholder.markdown(f"<div style='font-size:28px;font-weight:700;text-align:left;'>{density_percent}%</div>", unsafe_allow_html=True)
                status_placeholder.markdown(f"<div style='font-size:26px;font-weight:700;text-align:left;'>{status}</div>", unsafe_allow_html=True)
            except Exception:
                pass

            # update history chart
            try:
                if len(st.session_state.crowd_history) > 0:
                    # Use a simple line chart for sparkline-like history
                    history_chart.line_chart(st.session_state.crowd_history)
            except Exception:
                pass

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

            try:
                heat_color = generate_activation_cam(model, annotated, boxes=coords)
            except Exception:
                heat_color = generate_heatmap_from_boxes(annotated.shape, coords)

            # Convert BGR → RGB for Streamlit
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

            # Update placeholders side-by-side
            vid_placeholder.image(annotated_rgb, channels="RGB")
            heat_placeholder.image(heat_rgb, channels="RGB")

            # Store latest frames as PNG bytes for download buttons
            try:
                ok_a, buf_a = cv2.imencode('.png', cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
                ok_h, buf_h = cv2.imencode('.png', cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))
                if ok_a:
                    st.session_state.last_annotated_png = buf_a.tobytes()
                if ok_h:
                    st.session_state.last_heat_png = buf_h.tobytes()
            except Exception:
                pass

            # allow Streamlit to handle other events (like Stop button)
            time.sleep(0.01)
    finally:
        cap.release()
        st.session_state.running = False
        st.success("Analysis complete.")

    # After processing finished, offer downloads and CSV export if history exists
    try:
        if len(st.session_state.crowd_history) > 0:
            # CSV export
            import io, csv
            csv_buf = io.StringIO()
            writer = csv.writer(csv_buf)
            writer.writerow(["timestamp", "count"])
            for t, c in zip(st.session_state.time_history, st.session_state.crowd_history):
                writer.writerow([t, c])
            csv_bytes = csv_buf.getvalue().encode('utf-8')
            csv_buf.close()

            try:
                csv_col.download_button("Download History (CSV)", data=csv_bytes, file_name="crowd_history.csv", mime="text/csv")
            except Exception:
                pass

        # Latest frame downloads
        if st.session_state.get('last_annotated_png') is not None:
            try:
                dl_latest_annot.download_button("Download Annotated Frame", data=st.session_state.last_annotated_png, file_name="latest_annotated.png", mime="image/png")
            except Exception:
                pass
        if st.session_state.get('last_heat_png') is not None:
            try:
                dl_latest_heat.download_button("Download Heatmap Frame", data=st.session_state.last_heat_png, file_name="latest_heatmap.png", mime="image/png")
            except Exception:
                pass
    except Exception:
        pass


# Run processing if session state requests it. Protect with try/except so
# exceptions are logged and don't cause a silent Streamlit shutdown.
if st.session_state.running:
    try:
        _process_stream()
    except Exception as e:
        # Log full traceback
        logging.exception("Unhandled exception in processing loop")
        st.error(f"An error occurred during processing: {e}. See 'streamlit_error.log' for details.")


    