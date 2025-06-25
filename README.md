# 🚗 Drowsiness Detection System using YOLOv8 💤

This repository contains a computer vision-based **Drowsiness Detection System** built using **YOLOv8 (You Only Look Once Version 8)**. The system is designed to monitor driver alertness in real-time by analyzing facial landmarks—especially the eyes and head posture—to detect signs of fatigue or drowsiness.

---

## 🔍 Features

- Real-time face and eye detection using trained YOLO model
- Alarm/alert system when drowsiness is detected
- Works with webcam or video file input
- Lightweight and optimized for real-time performance

---

## 📦 Dataset Included

A labeled dataset for training and testing is included in the repository under the `dataset/` directory. It contains annotated images of:
- Eye states (open/closed)
- Facial expressions relevant for drowsiness and yawning detection

---

## 🧠 Technologies Used

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (PyTorch)
- OpenCV for video and image processing
- Dlib / Mediapipe (optional) for facial landmarks
- Python 3.x

---

### 📁 Repository Structure

```text
├── dataset/             # Labeled dataset (images + annotations)
├── model/               # YOLOv8 weights and configuration
│   ├── eye/             # Eye detection model
│   └── yawn/            # Yawn detection model
├── utils/               # Utility functions (EAR, head pose, alerts)
├── examples/            # Demo images and videos
├── main.py              # Main script to run drowsiness detection
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/drowsiness-detection-yolov8.git
```
3. Run the Detection Script
```bash
python main.py
```
