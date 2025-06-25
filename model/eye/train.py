from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x.pt")  # build a new model from YAML

if __name__ == "__main__":
# Train the model
    results = model.train(data="../../datasets/eye/data.yaml", epochs=50 , imgsz=196, batch=0.95)