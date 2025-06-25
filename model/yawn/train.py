from ultralytics import YOLO

# Load a model
model = YOLO("./yolov8x.pt")  # build a new model from YAML


if __name__ == "__main__":
# Train the model
    results = model.train(data="C:../../datasets/yawn/data.yaml", epochs=10  , imgsz=196, batch=0.95)