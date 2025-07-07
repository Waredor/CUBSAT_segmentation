import os
from PIL import Image, ImageDraw
import json

# Настройки классов и цветов
CLASS_NAMES = {"FT": 0, "Engine": 1, "Solar Panel": 2}
CLASS_COLORS = {
    0: (0, 255, 0),    # Зеленый для FT
    1: (255, 0, 0),    # Красный для Engine
    2: (0, 0, 255)     # Синий для Solar Panel
}

# Ручное указание путей (измените эти строки при необходимости)
image_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/0026.tif"
annotation_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/0026.json"
output_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/test_output/0026.png"

def read_yolo_annotations(annotation_path):
    """Читает LabelMe аннотации из файла .json"""
    if not os.path.exists(annotation_path):
        print(f"⚠ Файл аннотаций {annotation_path} не найден.")
        return None
    else:
        with open(annotation_path) as f:
            data = json.load(f)
            annotations = []
            shapes = data['shapes']
            for el in shapes:
                class_name = el['label']
                points = el['points']
                polygon = []
                class_id = CLASS_NAMES[class_name]

                for el in points:
                    polygon.append((el[0], el[1]))

                annotations.append({
                    "class_id": class_id,
                    "polygon": polygon
                })

        return annotations

def draw_masks(image_path, annotation_path, output_path):
    """Отрисовывает маски на изображении без преобразований."""
    # Отладочный вывод
    print(f"Проверка пути изображения: {image_path}")
    print(f"Текущая рабочая директория: {os.getcwd()}")

    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"❌ Изображение {image_path} не найдено. Проверьте путь или права доступа.")
        return

    # Открытие изображения
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"❌ Ошибка при открытии изображения {image_path}: {e}")
        return

    # Создание прозрачного слоя для масок
    mask_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)

    # Чтение аннотаций
    annotations = read_yolo_annotations(annotation_path)
    if not annotations:
        print(f"⚠ Нет аннотаций для {image_path}.")
        img.save(output_path)
        return

    # Отрисовка масок и bounding box
    for ann in annotations:
        class_id = ann["class_id"]
        polygon = ann["polygon"]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Белый по умолчанию

        # Денормализация координат полигона
        if polygon and len(polygon) >= 2:
            # Отрисовка полигона с заливкой
            draw.polygon(polygon, fill=color + (128,))

    # Наложение маски на изображение
    result = Image.alpha_composite(img, mask_layer)
    result = result.convert("RGB")  # Конвертация в RGB для сохранения в PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"✅ Изображение с масками сохранено: {output_path}")

def main():
    # Установка рабочей директории
    os.chdir(os.path.dirname(image_path) or os.getcwd())

    # Проверка существования файлов
    if not os.path.exists(image_path):
        print(f"❌ Изображение {image_path} не найдено. Проверьте путь или права доступа.")
        return
    if not os.path.exists(annotation_path):
        print(f"❌ Файл аннотаций {annotation_path} не найден.")
        return

    draw_masks(image_path, annotation_path, output_path)

if __name__ == "__main__":
    main()