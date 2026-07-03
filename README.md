# Electronic Schematic Symbol Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-black.svg)

## Overview
This project aims to bridge the gap between traditional engineering drafting and modern digital workflows by developing a robust computer vision pipeline. Using a YOLOv8 object detection model, our system can identify and classify 17 foundational electronic components from hand-drawn circuit schematics.

This repository contains the data pipeline, training configurations, and backend/frontend scaffolding necessary to train the model and deploy it as a web application.

## Dataset
We are utilizing the **Digitize-HCD** dataset, which consists of 1,277 high-resolution (600 DPI) images of hand-drawn schematics created by various volunteers.
- **Classes:** 17 components (Capacitor, Resistor, Diode, Inductor, Transistor, OpAmp, Ground, Vcc, Switch, LED, Battery, Relay, Transformer, Crystal, Speaker, Motor, Antenna)
- **Format:** COCO JSON annotations, converted to YOLOv8 format via our custom pipeline.

## Features
- **Data Pipeline:** Automated downloading, image downsampling (for memory optimization), and COCO-to-YOLO annotation conversion.
- **Object Detection:** Powered by Ultralytics YOLOv8 for fast and accurate bounding box predictions.
- **Web Interface:** A sleek Next.js frontend interacting with a FastAPI backend to process user-uploaded schematic images in real-time.

## Project Structure
```text
.
├── docs/               # Project proposals and presentations
├── data/               # Organized YOLO dataset (images/ and labels/)
├── AI4ALL.ipynb        # Main Google Colab notebook for pipeline and training
└── README.md           # Project documentation
```

## Prerequisites
- Python 3.10+
- Node.js 18+ (for frontend development)
- Ultralytics YOLOv8 library

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anything70/AI4ALL.git
   cd AI4ALL
   ```

2. **Open the Notebook:**
   Upload `AI4ALL.ipynb` to Google Colab to access the data pipeline and training environment.