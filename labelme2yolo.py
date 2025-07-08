import os
import json
import glob
from PIL import Image

# === ПАРАМЕТРЫ ===
LABELME_DIR = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/" # Папка с JSON и изображениями
YOLO_LABELS_DIR = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/labels/train/"      # Куда сохранить YOLO .txt файлы

# === СЛОВАРЬ КЛАССОВ ===
# Названия классов должны точно соответствовать тем, что в JSON (регистр важен!)
class_map = {
    'FT': 0,
    'Engine': 1,
    'Solar Panel': 2
}

# === ПОДГОТОВКА ===
os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
json_files = glob.glob(os.path.join(LABELME_DIR, "*.json"))

# === ОСНОВНОЙ ЦИКЛ ===
for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Имя и путь к изображению
    image_filename = data.get('imagePath')
    image_path = os.path.join(LABELME_DIR, image_filename)

    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        continue

    with Image.open(image_path) as img:
        w, h = img.size

    yolo_lines = []
    for shape in data['shapes']:
        label = shape.get('label')
        points = shape.get('points', [])

        if label not in class_map:
            print(f"Пропущен неизвестный класс: '{label}' в {json_path}")
            continue

        if len(points) < 3:
            print(f"Пропущен объект с недостатком точек в {json_path}")
            continue

        class_id = class_map[label]

        # Нормализация координат
        norm_points = []
        for x, y in points:
            norm_x = round(x / w, 6)
            norm_y = round(y / h, 6)
            norm_points.extend([norm_x, norm_y])

        yolo_line = f"{class_id} " + " ".join(map(str, norm_points))
        yolo_lines.append(yolo_line)

    # Сохраняем результат
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    txt_path = os.path.join(YOLO_LABELS_DIR, base_name + ".txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines))

    print(f"Сконвертирован: {base_name}.txt")
