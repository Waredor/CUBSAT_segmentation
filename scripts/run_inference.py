import os
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np

# Настройки классов и цветов
CLASS_NAMES = {0: "FT", 1: "Engine", 2: "Solar Panel", 3: "background"}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0)
}

# Ручное указание путей
image_path = "D:/Python projects/CUBSAT_segmentation/inference/1.jpg"
model_path = "D:/Python projects/CUBSAT_segmentation/inference/yolo11n-seg_final.pt"
output_path = "D:/Python projects/CUBSAT_segmentation/inference/output/1_pred.png"

def get_yolo_predictions(image_path, model_path):
    """Получает предсказания от модели YOLOv11."""
    if not os.path.exists(image_path):
        print(f"Изображение {image_path} не найдено.")
        return None
    if not os.path.exists(model_path):
        print(f"Модель {model_path} не найдена.")
        return None

    try:
        # Загрузка модели YOLOv11
        model = YOLO(model_path)
        # Выполнение инференса
        results = model.predict(image_path, conf=0.5)  # conf можно настроить
        annotations = []

        # Обработка результатов
        for result in results:
            if result.masks is not None:
                for mask, cls in zip(result.masks.xy, result.boxes.cls):
                    class_id = int(cls)
                    # Получение координат полигона маски
                    polygon = [(float(x), float(y)) for x, y in mask]
                    annotations.append({
                        "class_id": class_id,
                        "polygon": polygon
                    })

        return annotations
    except Exception as e:
        print(f"Ошибка при выполнении инференса: {e}")
        return None

def draw_masks(image_path, annotations, output_path):
    """Отрисовывает маски на изображении."""
    # Отладочный вывод
    print(f"Проверка пути изображения: {image_path}")
    print(f"Текущая рабочая директория: {os.getcwd()}")

    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"Изображение {image_path} не найдено. Проверьте путь или права доступа.")
        return

    # Открытие изображения
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"Ошибка при открытии изображения {image_path}: {e}")
        return

    # Создание прозрачного слоя для масок
    mask_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)

    # Проверка наличия аннотаций
    if not annotations:
        print(f"Нет предсказанных масок для {image_path}.")
        img.save(output_path)
        return

    # Отрисовка масок
    for ann in annotations:
        class_id = ann["class_id"]
        polygon = ann["polygon"]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Белый по умолчанию

        # Отрисовка полигона с заливкой
        if polygon and len(polygon) >= 2:
            draw.polygon(polygon, fill=color + (128,))

    # Наложение маски на изображение
    result = Image.alpha_composite(img, mask_layer)
    result = result.convert("RGB")  # Конвертация в RGB для сохранения в PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"Изображение с масками сохранено: {output_path}")

def main():
    # Установка рабочей директории
    os.chdir(os.path.dirname(image_path) or os.getcwd())

    # Проверка существования файлов
    if not os.path.exists(image_path):
        print(f"Изображение {image_path} не найдено. Проверьте путь или права доступа.")
        return
    if not os.path.exists(model_path):
        print(f"Модель {model_path} не найдена.")
        return

    # Получение предсказаний от YOLOv11
    annotations = get_yolo_predictions(image_path, model_path)
    if annotations is None:
        return

    # Отрисовка масок
    draw_masks(image_path, annotations, output_path)

if __name__ == "__main__":
    main()