import os
import json
import base64
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from PIL import Image
import cv2

logging.basicConfig(
    format='%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s',
    level=logging.INFO,
    filename='labeling_log.txt',
    filemode='w',
    encoding='utf-8'
)

stream_handler = logging.StreamHandler()
rotating_file_handler = RotatingFileHandler(
    filename='labeling_log.txt',
    maxBytes=1048576,
    backupCount=3
)
stream_handler.setLevel(logging.INFO)
rotating_file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(filename)s[LINE:%(lineno)d]# %(levelname)-8s '
                              '[%(asctime)s] %(message)s')
stream_handler.setFormatter(formatter)
rotating_file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addHandler(rotating_file_handler)

def yolo_to_labelme(yolo_file: str, image_path: str,
                    output_dir: str, class_names: list) -> None:
    """
    Метод yolo_to_labelme() осуществляет конвертацию .txt аннотаций в формате YOLOv11
    в .json формат LabelMe
    Parameters:
        yolo_file (str): путь к .txt файлу с аннотациями
        image_path (str): путь к файлу изображения
        output_dir (str): путь к директории для сохранения .json аннотаций
        class_names (list): список с именами классов
    """
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    shapes = []
    with open(yolo_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            class_id = int(parts[0])
            points = list(map(float, parts[1:]))
            if len(points) % 2 != 0:
                continue

            pixel_points = []
            for i in range(0, len(points), 2):
                x = points[i] * img_width
                y = points[i + 1] * img_height
                pixel_points.append([x, y])

            shape = {
                "label": class_names[class_id],
                "points": pixel_points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode(".jpg", image)
    image_data = base64.b64encode(buffer).decode("utf-8")

    labelme_data = {
        "version": "5.2.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageHeight": img_height,
        "imageWidth": img_width
    }

    output_json = os.path.join(output_dir, Path(yolo_file).stem + ".json")
    with open(output_json, 'w') as f:
        json.dump(labelme_data, f, indent=2)

    logger.info(f"Успешно сконвертирован файл {output_json}")

if __name__ == '__main__':
    labels_path = 'C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/test_output/yolo_labels/'
    output_path = 'C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/test_output/new_json_labels/'
    class_names = ["FT", "Engine", "Solar Panel"]
    logger.info("Начало работы")
    labels = [Path(os.path.join(labels_path, f)) for f in os.listdir(labels_path) if
                   f.endswith('.txt')]
    for label_path in labels:
        image_path = str(label_path.with_suffix(".jpg"))
        label_path = str(label_path)
        yolo_to_labelme(yolo_file=label_path, image_path=image_path,
                        output_dir=output_path, class_names=class_names)

    logger.info("Завершение работы")