🚗 Drowsiness Detection System using YOLOv8 💤
This repository contains a computer vision-based Drowsiness Detection System built using YOLOv8 (You Only Look Once Version 8). The system is designed to monitor driver alertness in real-time by analyzing facial landmarks, especially the eyes and head posture, to detect signs of fatigue or drowsiness.

🔍 Features
Real-time face and eye detection using YOLOv8

Eye Aspect Ratio (EAR) and blink rate analysis

Head pose estimation for detecting inattentiveness

Alarm/alert system when drowsiness is detected

Works with webcam or video file input

Lightweight and optimized for real-time performance

📦 Dataset Included
A labeled dataset for training and testing is included in the repository under the dataset/ directory. It contains annotated images of various eye states (open/closed) and facial expressions relevant for drowsiness detection.

🧠 Technologies Used
Ultralytics YOLOv8 (PyTorch)

OpenCV for image processing

Dlib / Mediapipe (optional) for facial landmarks

Python 3.x

📁 Repository Structure
bash
Copy
Edit
├── dataset/             # Labeled dataset (images + annotations)
├── model/               # YOLOv8 weights and config
├── utils/               # Utility functions (EAR, head pose, alerts)
├── examples/            # Demo images/videos
├── main.py              # Main script to run the detection
├── requirements.txt     # Python dependencies
└── README.md
🚀 Getting Started
Clone the repo: git clone https://github.com/your-username/drowsiness-detection-yolov8.git

Install requirements: pip install -r requirements.txt

Run the script: python main.py
