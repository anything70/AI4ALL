import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Electronic Schematic Symbol Classifier",
    page_icon="⚡",
    layout="wide"
)

# --- App Header ---
st.title("⚡ Electronic Schematic Symbol Classifier")
st.markdown("""
Welcome to the AI4ALL Schematic Classifier! 
Upload a picture of a hand-drawn electronic circuit schematic, and our YOLOv8 model will automatically detect and classify the components.
""")

# --- Load Model ---
@st.cache_resource
def load_model():
    # Load the ONNX model exported from Colab
    return YOLO('best.onnx', task='detect')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'best.onnx' is in the root directory.")
    st.stop()

# --- File Uploader ---
st.markdown("### Upload a Schematic")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        
    with st.spinner("Analyzing schematic..."):
        # Run YOLO inference
        results = model(image)
        
        # Plot the bounding boxes on the image
        # results[0].plot() returns a BGR numpy array
        annotated_image_bgr = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        
    with col2:
        st.subheader("Detected Components")
        st.image(annotated_image_rgb, use_column_width=True)
        
    # --- Analytics & Results ---
    st.markdown("### Detection Summary")
    
    # Get the predicted classes
    predicted_classes = results[0].boxes.cls.cpu().numpy()
    names = model.names
    
    if len(predicted_classes) == 0:
        st.warning("No components were detected in this image. Try uploading a clearer schematic.")
    else:
        # Count occurrences of each class
        class_counts = {}
        for cls_id in predicted_classes:
            class_name = names[int(cls_id)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # Display the counts cleanly
        st.write(f"**Total components detected: {len(predicted_classes)}**")
        
        # Create a nice layout for the metrics
        cols = st.columns(min(4, len(class_counts)))
        col_idx = 0
        for class_name, count in sorted(class_counts.items()):
            cols[col_idx % len(cols)].metric(label=class_name, value=count)
            col_idx += 1
