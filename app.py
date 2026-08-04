# -*- coding: utf-8 -*-
import streamlit as st
import onnxruntime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Electronic Schematic Symbol Classifier",
    page_icon="⚡",
    layout="wide"
)

LOCAL_ONNX_MODEL_PATH = "best.onnx"
IMG_SIZE = (640, 640)

CLASS_NAMES = [
    'BJT-NPN', 'BJT-PNP', 'Capacitor', 'Diode', 'GND', 'I-AC', 'I-DC', 
    'Inductor', 'MOSFET-N', 'MOSFET-P', 'Op-Amp', 'Resistor', 'V-AC', 
    'V-DC', 'V-DC (one port)', 'Wire Crossover', 'Zener Diode'
]

CLASS_COLORS = {
    'Resistor': (0, 102, 255),      # Bright Blue
    'Capacitor': (220, 20, 60),     # Crimson Red
    'Diode': (255, 140, 0),         # Dark Orange
    'GND': (0, 153, 76),            # Forest Green
    'Op-Amp': (138, 43, 226),       # Blue Violet
    'Inductor': (255, 20, 147),     # Deep Pink
    'BJT-NPN': (0, 206, 209),       # Dark Turquoise
    'BJT-PNP': (255, 215, 0),       # Gold
    'MOSFET-N': (75, 0, 130),       # Indigo
    'MOSFET-P': (128, 0, 128)       # Purple
}
DEFAULT_COLOR = (255, 0, 0)

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
def preprocess_image(image: Image.Image):
    img_resized = image.resize(IMG_SIZE)
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

def postprocess_output(output, original_size, conf_thresh=0.25, iou_thresh=0.45):
    predictions = np.squeeze(output[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    mask = scores > conf_thresh
    predictions = predictions[mask]
    scores = scores[mask]

    if len(predictions) == 0:
        return []

    boxes = predictions[:, :4]
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    orig_w, orig_h = original_size
    scale_x = orig_w / IMG_SIZE[0]
    scale_y = orig_h / IMG_SIZE[1]

    x_center, y_center, box_w, box_h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x_center - box_w / 2) * scale_x
    y1 = (y_center - box_h / 2) * scale_y
    x2 = (x_center + box_w / 2) * scale_x
    y2 = (y_center + box_h / 2) * scale_y
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
                if iou > iou_thresh:
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
st.sidebar.header("⚙️ Model Sensitivity Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
iou_threshold = st.sidebar.slider("Overlap Sensitivity (IoU)", 0.10, 0.90, 0.45, 0.05)

# --- File / Sample Image Selector ---
st.markdown("### Choose or Upload a Schematic")

# 1. Look for sample images in the "samples" directory
SAMPLE_DIR = "samples"
sample_files = []
if os.path.exists(SAMPLE_DIR):
    sample_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# 2. Add an input method choice
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
            input_tensor = preprocess_image(image)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            outputs = session.run([output_name], {input_name: input_tensor})

            detections = postprocess_output(outputs, orig_size, conf_threshold, iou_threshold)

            if detections:
                img_draw = image.copy()
                draw = ImageDraw.Draw(img_draw)
                font_size = max(12, int(0.02 * min(orig_size)))
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()

                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    label = f"{det['class_name']} {det['score']:.2f}"
                    color = CLASS_COLORS.get(det['class_name'], DEFAULT_COLOR)

                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.rectangle([x1, y1 - th - 6, x1 + tw + 6, y1], fill=color)
                    draw.text((x1 + 3, y1 - th - 3), label, fill="white", font=font)

                st.image(img_draw, use_container_width=True)
            else:
                st.warning("No components detected with current thresholds.")

        # Detection Metrics Summary
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


if input_type == "Use Sample Schematic":
    if sample_files:
        selected_sample = st.selectbox("Choose a sample schematic:", sample_files)
        sample_path = os.path.join(SAMPLE_DIR, selected_sample)
        image = Image.open(sample_path).convert("RGB")
        display_image(image)
    else:
        st.info("No sample images found in the 'samples/' folder. Please upload an image below.")

if input_type == "Upload Custom Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        display_image(image)

