import os
from PIL import Image, ImageDraw

# Настройки классов и цветов
CLASS_NAMES = {0: "Engine", 1: "FT", 2: "Solar Panel"}
CLASS_COLORS = {
    0: (255, 0, 0),    # Красный для Engine
    1: (0, 255, 0),    # Зеленый для FT
    2: (0, 0, 255)     # Синий для Solar Panel
}

# Ручное указание путей (измените эти строки при необходимости)
image_path = "D:/Python projects/CUBSAT_dataset_segmentation/0441.tif"
annotation_path = "D:/Python projects/CUBSAT_dataset_segmentation/labels/0441.txt"
output_path = "D:/Python projects/CUBSAT_dataset_segmentation/test_output/0441.png"

def read_yolo_annotations(annotation_path):
    """Читает YOLO аннотации из файла .txt."""
    annotations = []
    if not os.path.exists(annotation_path):
        print(f"⚠ Файл аннотаций {annotation_path} не найден.")
        return annotations

    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:  # Минимум: class_id + bbox (4 значения)
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                # Чтение полигона без преобразований
                polygon = []
                for i in range(5, len(parts), 2):
                    if i + 1 < len(parts):
                        x, y = float(parts[i]), float(parts[i + 1])
                        polygon.append((x, y))
                if polygon or len(parts) == 5:  # Поддержка как с полигоном, так и без
                    annotations.append({
                        "class_id": class_id,
                        "bbox": (x_center, y_center, width, height),
                        "polygon": polygon
                    })
            except (ValueError, IndexError) as e:
                print(f"⚠ Ошибка при парсинге строки в {annotation_path}: {e}")
                continue
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

    # Размеры изображения
    img_width, img_height = img.size

    # Отрисовка масок и bounding box
    for ann in annotations:
        class_id = ann["class_id"]
        polygon = ann["polygon"]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))  # Белый по умолчанию

        # Денормализация координат полигона
        if polygon and len(polygon) >= 2:
            denorm_polygon = [(x * img_width, y * img_height) for x, y in polygon]
            # Отрисовка полигона с заливкой
            draw.polygon(denorm_polygon, fill=color + (128,))

        # Отрисовка bounding box
        x_center, y_center, width, height = ann["bbox"]
        x1 = max(0, (x_center - width / 2) * img_width)
        y1 = max(0, (y_center - height / 2) * img_height)
        x2 = min(img_width, (x_center + width / 2) * img_width)
        y2 = min(img_height, (y_center + height / 2) * img_height)
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=2)

        # Добавление текста с названием класса
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        draw.text((x1 + 4, y1 + 4), class_name, fill=color + (255,))

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