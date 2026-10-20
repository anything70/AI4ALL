# -*- coding: utf-8 -*-
import streamlit as st
import onnxruntime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import io
import json
import yaml

# --- Page Configuration ---
st.set_page_config(
    page_title="Electronic Schematic Symbol Classifier",
    page_icon="⚡",
    layout="wide"
)

# --- Paths (anchored to this script's location, not the working directory) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ONNX_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best.onnx")
FONT_PATH = os.path.join(BASE_DIR, "arial.ttf")
SAMPLE_DIR = os.path.join(BASE_DIR, "samples")
CLASSES_YAML_PATH = os.path.join(BASE_DIR, "classes.yaml")

IMG_SIZE = (640, 640)
DEFAULT_COLOR = (255, 0, 0)


@st.cache_data(show_spinner=False)
def load_classes(path):
    """Load class names/colors from the single source of truth shared with
    the training notebook, instead of keeping a second hardcoded copy here
    that can silently drift out of sync with the model's actual class order.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    names = config["names"]
    colors = {name: tuple(rgb) for name, rgb in config.get("colors", {}).items()}
    return names, colors


try:
    CLASS_NAMES, CLASS_COLORS = load_classes(CLASSES_YAML_PATH)
except Exception as e:
    st.error(f"Couldn't load '{CLASSES_YAML_PATH}': {e}")
    st.stop()

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(LOCAL_ONNX_MODEL_PATH):
        st.error(f"ONNX model '{LOCAL_ONNX_MODEL_PATH}' not found locally.")
        return None
    try:
        session = onnxruntime.InferenceSession(LOCAL_ONNX_MODEL_PATH)
        return session
    except Exception as e:
        st.error(f"Error loading ONNX model: {e}")
        return None

session = load_model()

if session is None:
    st.stop()

# --- Helper Functions ---
def letterbox_resize(image, target_size=(640, 640), fill_color=(114, 114, 114)):
    """Resize an image to target_size while preserving aspect ratio, padding with fill_color."""
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h))

    new_image = Image.new("RGB", target_size, fill_color)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    new_image.paste(resized, (pad_x, pad_y))
    return new_image, scale, pad_x, pad_y


def preprocess_image(image: Image.Image):
    img_letterboxed, scale, pad_x, pad_y = letterbox_resize(image, IMG_SIZE)
    img_np = np.array(img_letterboxed).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np, scale, pad_x, pad_y


@st.cache_data(show_spinner=False)
def run_inference(_session, image_bytes: bytes, orig_size):
    """Run the ONNX forward pass once per distinct image and cache the raw
    output. Postprocessing (which depends on the confidence/IoU sliders) is
    done separately on every rerun so dragging a slider no longer re-triggers
    the expensive model forward pass.

    `_session` is prefixed with an underscore so Streamlit doesn't try to hash
    it; `image_bytes` (the actual cache key) captures everything that affects
    the model output.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor, scale, pad_x, pad_y = preprocess_image(image)
    input_name = _session.get_inputs()[0].name
    output_name = _session.get_outputs()[0].name
    outputs = _session.run([output_name], {input_name: input_tensor})
    return outputs, scale, pad_x, pad_y


def postprocess_output(output, scale, pad_x, pad_y, orig_w, orig_h, conf_thresh=0.25, iou_thresh=0.45):
    predictions = np.squeeze(output[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    mask = scores > conf_thresh
    predictions = predictions[mask]
    scores = scores[mask]

    if len(predictions) == 0:
        return []

    boxes = predictions[:, :4]
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    x_center, y_center, box_w, box_h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x_center - box_w / 2 - pad_x) / scale
    y1 = (y_center - box_h / 2 - pad_y) / scale
    x2 = (x_center + box_w / 2 - pad_x) / scale
    y2 = (y_center + box_h / 2 - pad_y) / scale

    x1 = np.clip(x1, 0, orig_w)
    x2 = np.clip(x2, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    indices = []
    if len(boxes) > 0:
        order = scores.argsort()[::-1]
        keep = [True] * len(boxes)
        for i in range(len(boxes)):
            idx_i = order[i]
            if not keep[idx_i]:
                continue
            indices.append(idx_i)
            for j in range(i + 1, len(boxes)):
                idx_j = order[j]
                if not keep[idx_j]:
                    continue
                ix1, iy1 = np.maximum(x1[idx_i], x1[idx_j]), np.maximum(y1[idx_i], y1[idx_j])
                ix2, iy2 = np.minimum(x2[idx_i], x2[idx_j]), np.minimum(y2[idx_i], y2[idx_j])
                iw, ih = np.maximum(0., ix2 - ix1), np.maximum(0., iy2 - iy1)
                inter = iw * ih
                area_i = (x2[idx_i] - x1[idx_i]) * (y2[idx_i] - y1[idx_i])
                area_j = (x2[idx_j] - x1[idx_j]) * (y2[idx_j] - y1[idx_j])
                union = area_i + area_j - inter
                iou = inter / union if union > 0 else 0
                if iou > iou_thresh and class_ids[idx_i] == class_ids[idx_j]:
                    keep[idx_j] = False

    return [{
        'box': boxes[idx].tolist(),
        'score': float(scores[idx]),
        'class_name': CLASS_NAMES[int(class_ids[idx])]
    } for idx in indices]


# --- UI Interface ---
st.title("⚡ Electronic Schematic Symbol Classifier")
st.markdown("Upload a picture of a hand-drawn electronic circuit schematic to automatically detect components.")

# Sidebar Settings
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45

if "conf_threshold" not in st.session_state:
    st.session_state.conf_threshold = DEFAULT_CONF_THRESHOLD
if "iou_threshold" not in st.session_state:
    st.session_state.iou_threshold = DEFAULT_IOU_THRESHOLD


def reset_thresholds():
    st.session_state.conf_threshold = DEFAULT_CONF_THRESHOLD
    st.session_state.iou_threshold = DEFAULT_IOU_THRESHOLD


st.sidebar.header("⚙️ Model Sensitivity Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.05, 0.95, step=0.05, key="conf_threshold"
)
iou_threshold = st.sidebar.slider(
    "Overlap Sensitivity (IoU)", 0.10, 0.90, step=0.05, key="iou_threshold"
)
st.sidebar.button("Reset to defaults", on_click=reset_thresholds)

# --- File / Sample Image Selector ---
st.markdown("### Choose or Upload a Schematic")

sample_files = []
if os.path.exists(SAMPLE_DIR):
    sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

input_type = st.radio(
    "Select Input Source:",
    ["Upload Custom Image", "Use Sample Schematic"],
    horizontal=True
)


# --- Process and Display ---
def display_image(image):
    if image is not None:
        orig_size = image.size

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detected Components")

            img_buf = io.BytesIO()
            image.save(img_buf, format="PNG")
            image_bytes = img_buf.getvalue()

            with st.spinner("Running detection..."):
                outputs, scale, pad_x, pad_y = run_inference(session, image_bytes, orig_size)

            detections = postprocess_output(
                outputs, scale, pad_x, pad_y,
                orig_size[0], orig_size[1],
                conf_threshold, iou_threshold
            )

            if detections:
                img_draw = image.copy()
                draw = ImageDraw.Draw(img_draw)
                font_size = max(12, int(0.02 * min(orig_size)))
                try:
                    font = ImageFont.truetype(FONT_PATH, font_size)
                except IOError:
                    # Pillow >= 9.2 supports a size arg on the default font so labels
                    # still scale with image resolution even without arial.ttf present.
                    try:
                        font = ImageFont.load_default(size=font_size)
                    except TypeError:
                        font = ImageFont.load_default()

                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    label = f"{det['class_name']} {det['score']:.2f}"
                    color = CLASS_COLORS.get(det['class_name'], DEFAULT_COLOR)

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    label_y = max(0, y1 - th - 6)
                    label_x2 = min(image.width, x1 + tw + 6)
                    draw.rectangle([x1, label_y, label_x2, label_y + th + 6], fill=color)
                    draw.text((x1 + 3, label_y + 3), label, fill="white", font=font)

                st.image(img_draw, use_container_width=True)

                buf = io.BytesIO()
                img_draw.save(buf, format="PNG")
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="Download annotated image",
                        data=buf.getvalue(),
                        file_name="detected_schematic.png",
                        mime="image/png"
                    )
                with dl_col2:
                    st.download_button(
                        label="Download detections (JSON)",
                        data=json.dumps(detections, indent=2),
                        file_name="detections.json",
                        mime="application/json"
                    )
            else:
                st.warning("No components detected with current thresholds.")

        if detections:
            st.markdown("### Detection Summary")
            st.write(f"**Total components detected: {len(detections)}**")

            counts = {}
            for det in detections:
                name = det['class_name']
                counts[name] = counts.get(name, 0) + 1

            cols = st.columns(min(4, len(counts)))
            for i, (name, count) in enumerate(sorted(counts.items())):
                cols[i % len(cols)].metric(label=name, value=count)


def safe_load_image(source):
    """Open an image and guard against corrupted files, unsupported formats,
    and unreasonably small/large uploads that would produce degenerate
    letterbox results or slow the app down."""
    try:
        image = Image.open(source).convert("RGB")
    except Exception:
        st.error("Couldn't read that image — try a different file.")
        st.stop()

    w, h = image.size
    if w < 100 or h < 100:
        st.warning(f"That image is only {w}x{h}px — detections on very small images tend to be unreliable.")
    elif w > 8000 or h > 8000:
        st.warning(f"That image is {w}x{h}px — very large images may be slow to process.")

    return image


if input_type == "Use Sample Schematic":
    if sample_files:
        selected_sample = st.selectbox("Choose a sample schematic:", sample_files)
        sample_path = os.path.join(SAMPLE_DIR, selected_sample)
        image = safe_load_image(sample_path)
        display_image(image)
    else:
        st.info("No sample images found in the 'samples/' folder. Please upload an image below.")

if input_type == "Upload Custom Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = safe_load_image(uploaded_file)
        display_image(image)
