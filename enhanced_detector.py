import cv2
import logging
import time
import winsound
import os
import threading
import queue
import yaml
from ultralytics import YOLO
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2

@dataclass
class DetectionConfig:
    # Model paths
    eye_model_path: str
    yawn_model_path: str
    
    # Detection thresholds
    eye_close_threshold: int = 100
    yawn_threshold: int = 90
    critical_yawn_threshold: int = 120
    
    # Performance settings
    frame_skip: int = 2
    gpu_acceleration: bool = True
    confidence_threshold: float = 0.5
    
    # Alert settings
    alert_frequency: int = 1000  # Hz
    alert_duration: int = 100    # ms

class DrowsinessDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False
        
        # Initialize models
        self.eye_model = self._load_model(config.eye_model_path)
        self.yawn_model = self._load_model(config.yawn_model_path)
        
        # State variables
        self.eye_closure_count = 0
        self.yawn_count = 0
        self.frame_count = 0
        self.tired_level = 0
        
        # Threads
        self.capture_thread = None
        self.processing_thread = None
        
    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLO model with GPU acceleration if available."""
        try:
            model = YOLO(model_path)
            if self.config.gpu_acceleration:
                model.to('cuda' if self._has_gpu() else 'cpu')
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _capture_frames(self):
        """Capture frames from webcam in a separate thread."""
        # Try to use DirectShow on Windows for better performance
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except:
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
            
        try:
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            # Get actual camera FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default to 30 FPS if can't detect
            
            frame_time = 1.0 / fps
            last_time = time.time()
            
            while self.running:
                current_time = time.time()
                
                # Skip frame if processing is running behind
                if current_time - last_time < frame_time:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                    
                last_time = current_time
                
                # Grab a frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 480))
                
                # Put frame in queue if not full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                self.frame_count += 1
                
        except Exception as e:
            logger.error(f"Error in capture thread: {e}")
        finally:
            cap.release()
    
    def _process_frames(self):
        """Process frames in a separate thread."""
        import time
        import numpy as np
        
        # Warm up models with a blank frame
        try:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.eye_model(dummy_frame, verbose=False)
            self.yawn_model(dummy_frame, verbose=False)
            logger.info("Models warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
            # Continue anyway as this is just warmup
        
        fps_counter = 0
        last_fps_time = time.time()
        
        while self.running or not self.frame_queue.empty():
            try:
                start_time = time.time()
                
                # Get frame with timeout
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    time.sleep(0.001)
                    continue
                
                # Process eye detection with half precision for speed
                eye_results = self.eye_model.track(
                    frame, 
                    conf=self.config.confidence_threshold,
                    verbose=False,
                    half=True,  # Use half precision for faster inference
                    device='0' if self.config.gpu_acceleration and self._has_gpu() else 'cpu',
                    max_det=1  # Limit to 1 detection per class for speed
                )
                
                # Process yawn detection with same optimizations
                yawn_results = self.yawn_model.track(
                    frame,
                    conf=self.config.confidence_threshold,
                    verbose=False,
                    half=True,
                    device='0' if self.config.gpu_acceleration and self._has_gpu() else 'cpu',
                    max_det=1
                )
                
                # Calculate FPS
                fps_counter += 1
                if time.time() - last_fps_time >= 1.0:  # Update FPS every second
                    self.current_fps = fps_counter / (time.time() - last_fps_time)
                    fps_counter = 0
                    last_fps_time = time.time()
                
                # Update state
                self._update_detection_state(eye_results, yawn_results)
                
                # Get alert level
                alert_level = self._get_alert_level()
                
                # Put results in queue
                if not self.result_queue.full():
                    self.result_queue.put({
                        'frame': frame,
                        'eye_results': eye_results,
                        'yawn_results': yawn_results,
                        'alert_level': alert_level,
                        'eye_closure_count': self.eye_closure_count,
                        'yawn_count': self.yawn_count,
                        'tired_level': self.tired_level
                    })
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
    
    def _update_detection_state(self, eye_results, yawn_results):
        """Update detection state based on model results."""
        # Update eye closure count
        eye_closed = any(int(box.cls) == 0 for r in eye_results for box in r.boxes)
        if eye_closed:
            self.eye_closure_count += 1
        else:
            self.eye_closure_count = max(0, self.eye_closure_count - 5)
        
        # Update yawn count
        yawn_detected = any(len(r.boxes) > 0 for r in yawn_results)
        if yawn_detected:
            self.yawn_count += 1
        
        # Update tired level
        if self.yawn_count > self.config.yawn_threshold:
            self.tired_level = min(5, self.tired_level + 1)
        
        # Reset counters periodically
        if self.frame_count % 1800 == 0:  # Reset every ~60 seconds at 30 FPS
            self.yawn_count = 0
        if self.frame_count % 9000 == 0:  # Reset tired level every ~5 minutes
            self.tired_level = 0
    
    def _get_alert_level(self) -> AlertLevel:
        """Determine the current alert level."""
        if (self.eye_closure_count > self.config.eye_close_threshold or 
            self.yawn_count > self.config.critical_yawn_threshold or
            self.tired_level >= 4):
            return AlertLevel.CRITICAL
        elif (self.yawn_count > self.config.yawn_threshold or 
              self.tired_level >= 2):
            return AlertLevel.WARNING
        return AlertLevel.NORMAL
    
    def _trigger_alert(self, level: AlertLevel):
        """Trigger appropriate alert based on alert level."""
        if level == AlertLevel.CRITICAL:
            winsound.Beep(self.config.alert_frequency, self.config.alert_duration)
            logger.warning("CRITICAL: Take a break immediately!")
        elif level == AlertLevel.WARNING:
            logger.warning("WARNING: You seem tired. Consider taking a break.")
    
    def start(self):
        """Start the detection system."""
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
        logger.info("Drowsiness detection system started")
    
    def stop(self):
        """Stop the detection system."""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
            
        logger.info("Drowsiness detection system stopped")
    
    def run(self):
        """Run the main detection loop."""
        self.start()
        last_alert_time = 0
        alert_cooldown = 2  # Reduced from 5 seconds for more responsive alerts
        fps_update_time = time.time()
        frame_count = 0
        self.current_fps = 0
        
        try:
            while self.running:
                try:
                    result = self.result_queue.get(timeout=1)
                    
                    # Get results
                    frame = result['frame']
                    eye_results = result['eye_results']
                    yawn_results = result['yawn_results']
                    
                    # Simple plotting for better performance
                    for r in eye_results + yawn_results:
                        if hasattr(r, 'boxes'):
                            for box in r.boxes.xyxy.cpu().numpy():
                                x1, y1, x2, y2 = map(int, box[:4])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Show FPS
                    if getattr(self.config, 'show_fps', True):
                        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', 
                                 (10, frame.shape[0] - 10), 
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Display status
                    status = (f"Eyes: {result['eye_closure_count']} | "
                             f"Yawns: {result['yawn_count']} | "
                             f"Tired: {'▉' * result['tired_level']}")
                    
                    cv2.putText(frame, status, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show alert if needed
                    current_time = time.time()
                    if (result['alert_level'] != AlertLevel.NORMAL and 
                        current_time - last_alert_time > alert_cooldown):
                        self._trigger_alert(result['alert_level'])
                        last_alert_time = current_time
                    
                    # Display the frame
                    cv2.imshow('Drowsiness Detection', frame)
                    
                    # Break on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop()
            cv2.destroyAllWindows()

def load_config(config_path: str = 'config.yaml') -> DetectionConfig:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Only keep the parameters that DetectionConfig expects
        valid_params = {
            'eye_model_path', 'yawn_model_path', 'eye_close_threshold', 
            'yawn_threshold', 'critical_yawn_threshold', 'frame_skip',
            'gpu_acceleration', 'confidence_threshold', 'alert_frequency',
            'alert_duration', 'display_fps', 'display_status', 'log_level'
        }
        
        # Filter config_data to only include valid parameters
        filtered_config = {k: v for k, v in config_data.items() if k in valid_params}
        
        # Set default values for required parameters if not provided
        if 'eye_model_path' not in filtered_config:
            filtered_config['eye_model_path'] = os.path.join("model", "eye", "runs", "detect", "train3", "weights", "best.pt")
        if 'yawn_model_path' not in filtered_config:
            filtered_config['yawn_model_path'] = os.path.join("model", "yawn", "runs", "detect", "train", "weights", "best.pt")
            
        return DetectionConfig(**filtered_config)
        
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using default settings.")
        return DetectionConfig(
            eye_model_path=os.path.join("model", "eye", "runs", "detect", "train3", "weights", "best.pt"),
            yawn_model_path=os.path.join("model", "yawn", "runs", "detect", "train", "weights", "best.pt")
        )

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Create and run detector
    detector = DrowsinessDetector(config)
    detector.run()
