from ultralytics import YOLO

# Create YOLOv11n model for instance segmentation
model_cfg = "Model_cfg/yolo11n-seg.yaml"
data_cfg = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/dataset.yaml"
model = YOLO(model_cfg)

# Training model
model.train(data=data_cfg,
            epochs=50,
            imgsz=640,
            batch=4,
            patience=30,
            device="cpu"
)

# Validation
metrics = model.val()
print(f"mAP50-95 (masks): {metrics.seg.map}")

# Inference
results = model.predict("path/to/image.jpg", save=True)

# Results processing
for result in results:
    # Masks and BBoxes
    masks = result.masks.xy
    boxes = result.boxes.xyxy
    classes = result.boxes.cls
    for mask, box, cls in zip(masks, boxes, classes):
        print(f"Class: {cls}, Box: {box}, Mask: {mask}")