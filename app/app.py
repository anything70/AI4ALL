# -*- coding: utf-8 -*-
import ast
import hashlib
import io
import json
import os

import numpy as np
import onnxruntime
import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

# --- Page Configuration ---
st.set_page_config(
    page_title="Electronic Schematic Symbol Classifier",
    page_icon="⚡",
    layout="wide"
)

# --- Paths (anchored to this script's location, not the working directory) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ONNX_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best.onnx")
SAMPLE_DIR = os.path.join(BASE_DIR, "samples")
CLASSES_YAML_PATH = os.path.join(BASE_DIR, "classes.yaml")

DEFAULT_IMG_SIZE = (640, 640)
DEFAULT_COLOR = (255, 0, 0)
MAX_DETECTIONS = 300


@st.cache_data(show_spinner=False)
def load_classes(path):
    """Load class names/colors from classes.yaml. The names here are only a
    fallback — the model's own embedded metadata wins when it's available."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    names = config["names"]
    colors = {name: tuple(rgb) for name, rgb in config.get("colors", {}).items()}
    return names, colors


try:
    YAML_CLASS_NAMES, CLASS_COLORS = load_classes(CLASSES_YAML_PATH)
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


def read_model_metadata(session, fallback_names):
    """Ultralytics embeds the class names and input size in the exported ONNX
    file, so reading them from there means labels can never drift out of sync
    with the weights that are actually loaded."""
    meta = session.get_modelmeta().custom_metadata_map
    try:
        names_dict = ast.literal_eval(meta["names"])
        names = [names_dict[i] for i in sorted(names_dict)]
    except (KeyError, ValueError, SyntaxError):
        names = fallback_names
    try:
        img_size = tuple(ast.literal_eval(meta["imgsz"]))
    except (KeyError, ValueError, SyntaxError):
        img_size = DEFAULT_IMG_SIZE
    return names, img_size


CLASS_NAMES, IMG_SIZE = read_model_metadata(session, YAML_CLASS_NAMES)

# --- Helper Functions ---
def letterbox_resize(image, target_size=(640, 640), fill_color=(114, 114, 114)):
    """Resize an image to target_size while preserving aspect ratio, padding with fill_color."""
    orig_w, orig_h = image.size
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

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
def run_inference(_session, _image, image_key: str):
    """Run the ONNX forward pass once per distinct image and cache the raw
    output. Postprocessing (which depends on the confidence/IoU sliders) is
    done separately on every rerun so dragging a slider doesn't re-trigger
    the expensive model forward pass.

    `_session` and `_image` are prefixed with an underscore so Streamlit
    doesn't try to hash them; `image_key` (a hash of the source file's bytes)
    is the actual cache key.
    """
    input_tensor, scale, pad_x, pad_y = preprocess_image(_image)
    input_name = _session.get_inputs()[0].name
    output_name = _session.get_outputs()[0].name
    outputs = _session.run([output_name], {input_name: input_tensor})
    return outputs, scale, pad_x, pad_y


def non_max_suppression(boxes, scores, class_ids, iou_thresh, max_det=MAX_DETECTIONS):
    """Greedy per-class NMS, vectorized with NumPy.

    Boxes are shifted by a per-class offset so boxes of different classes can
    never overlap, which lets a single pass handle every class at once.
    """
    offsets = class_ids[:, None] * (boxes.max() + 1)
    shifted = boxes + offsets
    x1, y1, x2, y2 = shifted.T
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thresh]
    return keep


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
    x1 = np.clip((x_center - box_w / 2 - pad_x) / scale, 0, orig_w)
    y1 = np.clip((y_center - box_h / 2 - pad_y) / scale, 0, orig_h)
    x2 = np.clip((x_center + box_w / 2 - pad_x) / scale, 0, orig_w)
    y2 = np.clip((y_center + box_h / 2 - pad_y) / scale, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    indices = non_max_suppression(boxes, scores, class_ids, iou_thresh)

    return [{
        'box': [round(float(v), 1) for v in boxes[idx]],
        'score': round(float(scores[idx]), 4),
        'class_name': CLASS_NAMES[int(class_ids[idx])]
    } for idx in indices]


def load_label_font(size):
    # Pillow >= 10.1 ships a scalable open-licensed default font, so labels
    # scale with image resolution without bundling a font file.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def text_color_for(rgb):
    """Black text on light label backgrounds (yellow, cyan), white on dark ones."""
    r, g, b = rgb
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "white"


def draw_detections(image, detections):
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    line_width = max(2, int(0.004 * min(image.size)))
    font = load_label_font(max(12, int(0.02 * min(image.size))))

    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = f"{det['class_name']} {det['score']:.2f}"
        color = CLASS_COLORS.get(det['class_name'], DEFAULT_COLOR)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y1 - th - 6)
        label_x1 = max(0, min(x1, image.width - tw - 6))
        draw.rectangle([label_x1, label_y, label_x1 + tw + 6, label_y + th + 6], fill=color)
        draw.text((label_x1 + 3, label_y + 3 - bbox[1]), label, fill=text_color_for(color), font=font)

    return img_draw


# --- UI Interface ---
st.title("⚡ Electronic Schematic Symbol Classifier")
st.markdown(
    "Upload a photo or scan of a **hand-drawn circuit schematic** and a YOLOv8 Nano "
    f"model will detect and label its components across **{len(CLASS_NAMES)} symbol classes**."
)

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
    "Confidence Threshold", 0.05, 0.95, step=0.05, key="conf_threshold",
    help="Only show detections the model is at least this confident about."
)
iou_threshold = st.sidebar.slider(
    "Overlap Sensitivity (IoU)", 0.10, 0.90, step=0.05, key="iou_threshold",
    help="Lower values merge overlapping boxes of the same component more aggressively."
)
st.sidebar.button("Reset to defaults", on_click=reset_thresholds, width="stretch")

with st.sidebar.expander("🔎 Detectable components"):
    st.markdown("\n".join(f"- {name}" for name in CLASS_NAMES))

# --- File / Sample Image Selector ---
st.markdown("### Choose or Upload a Schematic")

sample_files = []
if os.path.exists(SAMPLE_DIR):
    sample_files = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg')))

input_type = st.radio(
    "Select Input Source:",
    ["Upload Custom Image", "Use Sample Schematic"],
    horizontal=True
)


# --- Process and Display ---
def display_image(image, image_key):
    orig_w, orig_h = image.size

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Detected Components")

        with st.spinner("Running detection..."):
            outputs, scale, pad_x, pad_y = run_inference(session, image, image_key)

        detections = postprocess_output(
            outputs, scale, pad_x, pad_y,
            orig_w, orig_h,
            conf_threshold, iou_threshold
        )

        if not detections:
            st.warning("No components detected with current thresholds. Try lowering the confidence threshold.")
            return

        img_draw = draw_detections(image, detections)
        st.image(img_draw, width="stretch")

        buf = io.BytesIO()
        img_draw.save(buf, format="PNG")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇️ Annotated image",
                data=buf.getvalue(),
                file_name="detected_schematic.png",
                mime="image/png",
                width="stretch"
            )
        with dl_col2:
            st.download_button(
                label="⬇️ Detections (JSON)",
                data=json.dumps(detections, indent=2),
                file_name="detections.json",
                mime="application/json",
                width="stretch"
            )

    st.divider()
    st.markdown("### Detection Summary")

    counts = {}
    for det in detections:
        name = det['class_name']
        counts[name] = counts.get(name, 0) + 1

    st.metric(label="Total components detected", value=len(detections))
    cols = st.columns(min(4, len(counts)))
    for i, (name, count) in enumerate(sorted(counts.items())):
        cols[i % len(cols)].metric(label=name, value=count)


def safe_load_image(image_bytes):
    """Open an image and guard against corrupted files, unsupported formats,
    and unreasonably small/large uploads that would produce degenerate
    letterbox results or slow the app down."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Phone photos store rotation in EXIF rather than in the pixels.
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        st.error("Couldn't read that image — try a different file.")
        st.stop()

    w, h = image.size
    if w < 100 or h < 100:
        st.warning(f"That image is only {w}x{h}px — detections on very small images tend to be unreliable.")
    elif w > 8000 or h > 8000:
        st.warning(f"That image is {w}x{h}px — very large images may be slow to process.")

    return image


def show(image_bytes):
    image = safe_load_image(image_bytes)
    display_image(image, hashlib.sha256(image_bytes).hexdigest())


if input_type == "Use Sample Schematic":
    if sample_files:
        selected_sample = st.selectbox("Choose a sample schematic:", sample_files)
        with open(os.path.join(SAMPLE_DIR, selected_sample), "rb") as f:
            show(f.read())
    else:
        st.info("No sample images found in the 'samples/' folder. Please upload an image below.")

if input_type == "Upload Custom Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        show(uploaded_file.getvalue())
