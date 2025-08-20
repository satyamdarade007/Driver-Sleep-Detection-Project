# 🚗 Enhanced Driver Sleep Detection System using YOLOv8 💤

A high-performance, real-time drowsiness detection system that monitors driver alertness using computer vision and deep learning. The system analyzes eye states and yawning patterns to detect signs of fatigue, triggering alerts to prevent potential accidents.


## ✨ Key Features

- 🚀 **Real-time Processing**: Optimized for high FPS performance
- 👁️ **Multi-stage Detection**: Tracks both eye states and yawning
- 🔔 **Smart Alerts**: Progressive warning system based on fatigue levels
- ⚡ **GPU Acceleration**: Automatically utilizes CUDA if available
- 🛠️ **Configurable**: Easy to adjust detection parameters via YAML config
- 📊 **Visual Feedback**: Real-time FPS counter and status display

## 🛠️ Technologies

- **Core**: Python 3.8+
- **Computer Vision**: OpenCV, YOLOv8
- **Deep Learning**: PyTorch, Ultralytics
- **Audio Alerts**: winsound (Windows)
- **Configuration**: YAML

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install ultralytics opencv-python pyyaml
   ```

3. **Download pre-trained models**
   - Place your YOLOv8 models in the `model/` directory
   - Update the paths in `config.yaml` if needed

## 🚀 Usage

1. **Run the detection system**
   ```bash
   python enhanced_detector.py
   ```

2. **Keyboard Controls**
   - `Q`: Quit the application
   - `P`: Pause/Resume detection

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Detection thresholds
- Alert settings
- Performance parameters
- Display options

## 📂 Project Structure

```
.
├── config.yaml           # Configuration file
├── enhanced_detector.py  # Main application
├── requirements.txt      # Python dependencies
├── buzzer.mp3           # Alert sound
├── model/               # YOLO models
│   ├── eye/            # Eye state detection
│   └── yawn/           # Yawn detection
└── README.md           # This file
```

## 📈 Performance Tips

- Use a CUDA-enabled GPU for best performance
- Adjust `frame_skip` in `config.yaml` for different performance/accuracy tradeoffs
- Lower resolution in `_capture_frames()` for faster processing
- Close other GPU-intensive applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV community

