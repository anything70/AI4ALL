# Electronic Schematic Symbol Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)
![ONNX](https://img.shields.io/badge/ONNX-Model-blue.svg)

## Overview
This project aims to bridge the gap between traditional engineering drafting and modern digital workflows by developing a robust computer vision pipeline. Using a YOLOv8 Nano object detection model, our system can identify and classify 17 foundational electronic components from hand-drawn circuit schematics.

This repository contains the data pipeline, training configurations, and a Streamlit web application to serve the trained model for real-time inference.

## Dataset
We are utilizing the **Digitize-HCD** dataset, which consists of 1,277 high-resolution (600 DPI) images of hand-drawn schematics created by various volunteers.
- **Classes:** 17 components (Capacitor, Resistor, Diode, Inductor, Transistor, OpAmp, Ground, Vcc, Switch, LED, Battery, Relay, Transformer, Crystal, Speaker, Motor, Antenna)
- **Format:** COCO JSON annotations, converted to YOLOv8 format via our custom pipeline.

## Features
- **Data Pipeline:** Automated downloading, image downsampling (for memory optimization), and COCO-to-YOLO annotation conversion.
- **Object Detection:** Powered by Ultralytics YOLOv8 for fast and accurate bounding box predictions. We utilized robust data augmentations to generalize the model across varying handwriting styles.
- **Interactive Web Interface:** A seamless Streamlit application (`app_py.py`) utilizing the native Ultralytics engine for ONNX inference to process user-uploaded schematic images in real-time.

## Project Structure
```text
.
├── docs/                                          # Project proposals and presentations
├── datasets/                                      # Organized YOLO dataset
├── app_py.py                                      # Main Streamlit web application
├── best.onnx                                      # Exported YOLOv8 Nano model weights
├── YOLOv8_Schematic_Component_Detector...ipynb    # Main Google Colab notebook for training
├── requirements.txt                               # Python dependencies
└── README.md                                      # Project documentation
```

## Prerequisites
- Python 3.10+
- Streamlit
- ONNX Runtime
- Pillow, Numpy (see requirements.txt)

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anything70/AI4ALL.git
   cd AI4ALL
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   Start the local web server to test the model:
   ```bash
   streamlit run app_py.py
   ```
   Open your browser and navigate to the provided localhost URL (usually `http://localhost:8501`) to upload schematics and test the model.