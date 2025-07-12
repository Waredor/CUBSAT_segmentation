import base64
import os
import cv2
import numpy as np
import json
from ultralytics import YOLO
from pathlib import Path
import torch

# 1. Настройка путей и параметров
DATASET_PATH = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/"  # Укажите путь к вашему датасету
MODEL_PATH = "Model_cfg/yolo11n-seg_labeling.pt"  # Предобученная модель YOLOv11n для сегментации аннотаций
OUTPUT_DIR = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/"  # Папка для сохранения аннотаций LabelMe
NUM_CLASSES = 3  # Количество классов в вашем датасете
CLASS_NAMES = ["FT", "Engine", "Solar Panel"]  # Названия классов
IMG_SIZE = 1024  # Размер изображения для обучения и инференса
EPOCHS = 30  # Количество эпох для fine-tuning
BATCH_SIZE = 4  # Размер батча
LEARNING_RATE = 0.001  # Скорость обучения
FREEZE_LAYERS = 10


# 2. Создание YAML-файла для датасета
def create_yaml_file():
    yaml_content = f"""
path: {DATASET_PATH}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
    yaml_path = os.path.join(DATASET_PATH, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path

def freeze_layers(model, num_layers_to_freeze):
    modules = list(model.model.modules())
    frozen_layers = 0

    for module in modules:
        if frozen_layers >= num_layers_to_freeze:
            break
        for param in module.parameters():
            param.requires_grad = False
            frozen_layers += 1

    print(f"Заморожено {frozen_layers} слоев")

# 3. Fine-tuning модели
def fine_tune_model(yaml_path):
    model = YOLO(MODEL_PATH)
    model.model.yaml['imgsz'] = 1024
    # Замораживаем backbone для предотвращения переобучения
    freeze_layers(model, FREEZE_LAYERS)
    # Запускаем обучение
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        optimizer="Adam",
        patience=10,  # Early stopping после 10 эпох без улучшения
        device=0 if torch.cuda.is_available() else "cpu",
        augment=True  # Включаем аугментации
    )
    return model


# 4. Преобразование маски в полигоны
def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Полигон должен иметь минимум 3 точки
            polygon = contour.squeeze().tolist()
            polygons.append(polygon)
    return polygons


# 5. Создание JSON в формате LabelMe
def create_labelme_json(image_path, masks, labels, class_names, output_dir):
    image_name = Path(image_path).name
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    _, buffer = cv2.imencode(".jpg", image)
    image_data = base64.b64encode(buffer).decode("utf-8")


    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_name,
        "imageData": image_data,
        "imageHeight": height,
        "imageWidth": width
    }

    for mask, label in zip(masks, labels):
        polygons = mask_to_polygons(mask.astype(np.uint8))
        for polygon in polygons:
            shape = {
                "label": class_names[int(label)],
                "points": polygon,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme_data["shapes"].append(shape)

    output_path = os.path.join(output_dir, image_name.replace(".jpg", ".json").replace(".png", ".json"))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labelme_data, f, indent=2)
    return output_path


# 6. Инференс и конвертация в LabelMe
def run_inference_and_convert(model, test_images_dir, output_dir):
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if
                   f.endswith(('.jpg', '.png'))]

    for image_path in test_images:
        # Инференс
        results = model.predict(image_path, imgsz=IMG_SIZE, conf=0.5, iou=0.7)

        # Получение масок и меток
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # Маски объектов
            labels = results[0].boxes.cls.cpu().numpy()  # Классы объектов
            create_labelme_json(image_path, masks, labels, CLASS_NAMES, output_dir)
        else:
            print(f"Нет объектов в {image_path}")


# 7. Основной процесс
def main():
    # Создаем YAML-файл для датасета
    yaml_path = create_yaml_file()

    # Fine-tuning модели
    model = fine_tune_model(yaml_path)

    # Сохранение модели
    model.save("Model_cfg/yolo11n-seg_labeling.pt")

    # Инференс и конвертация в LabelMe
    test_images_dir = os.path.join(DATASET_PATH, "images/test")
    run_inference_and_convert(model, test_images_dir, OUTPUT_DIR)
    print(f"Аннотации в формате LabelMe сохранены в {OUTPUT_DIR}")

if __name__ == "__main__":
    import torch
    main()