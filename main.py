from ultralytics import YOLO

# Create YOLOv11n model for instance segmentation
model = YOLO("yolo11n-seg.yaml")

# Training model
model.train(data="",
            epochs=50,
            imgsz=640,
            batch=16,
            patience=30,
            device=0
)
