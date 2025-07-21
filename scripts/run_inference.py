import os
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Ручное указание путей
IMAGE_PATH = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/CUBSAT_segmentation/inference/1.jpg"
MODEL_PATH = ("C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/CUBSAT_segmentation/inference"
              "/yolo11n-seg_final.pt")
OUTPUT_PATH = ("C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/CUBSAT_segmentation/inference/"
               "output/1_pred.png")

logging.basicConfig(
    format='%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s',
    level=logging.INFO,
    filename='inference_log.txt',
    filemode='w',
    encoding='utf-8'
)

stream_handler = logging.StreamHandler()
rotating_file_handler = RotatingFileHandler(
    filename='inference_log.txt',
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


# Настройки классов и цветов
CLASS_NAMES = {0: "FT", 1: "Engine", 2: "Solar Panel", 3: "background"}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0)
}

def get_yolo_predictions(image_dir: str, model_dir: str) -> list:
    """
    Метод get_yolo_predictions()
    получает предсказания от модели YOLOv11.
    Parameters:
        image_dir (str): путь к изображению для инференса.
        model_dir(str): путь к файлу .pt модели YOLOv11.
    Returns:
        annotations (dict): Словарь с предсказанными моделью масками и метками классов.
    """
    if not os.path.exists(image_dir):
        logger.warning(f"Изображение {image_dir} не найдено.")

    if not os.path.exists(model_dir):
        logger.warning(f"Модель {model_dir} не найдена.")

    try:
        model = YOLO(model_dir)
        results = model.predict(image_dir, conf=0.5)
        annotations = []

        for result in results:
            if result.masks is not None:
                for mask, cls in zip(result.masks.xy, result.boxes.cls):
                    class_id = int(cls)
                    polygon = [(float(x), float(y)) for x, y in mask]
                    annotations.append({
                        "class_id": class_id,
                        "polygon": polygon
                    })

        return annotations
    except Exception as e:
        logger.error(f"Ошибка при выполнении инференса: {e}")
        raise RuntimeError(f"Ошибка при выполнении инференса: {e}")


def draw_masks(image_dir: str, annotations: list, output_dir: str):
    """
    Метод draw_masks()
    отрисовывает полигоны на изображении.
    Parameters:
        image_dir (str): путь к изображению для инференса.
        annotations (list): список с предсказанными моделью масками и метками классов.
        output_dir (str): путь к выходной директории для сохранения изображения
            с полигонами классов.
    """
    logger.info(f"Проверка пути изображения: {image_dir}")
    logger.info(f"Текущая рабочая директория: {os.getcwd()}")

    if not os.path.exists(image_dir):
        logger.warning(f"Изображение {image_dir} не найдено. Проверьте путь или права доступа.")
        return

    try:
        img = Image.open(image_dir).convert("RGBA")
    except Exception as e:
        logger.error(f"Ошибка при открытии изображения {image_dir}: {e}")
        return

    # Создание прозрачного слоя для масок
    mask_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)

    # Проверка наличия аннотаций
    if not annotations:
        logger.warning(f"Нет предсказанных масок для {image_dir}.")
        img.save(output_dir)
        return

    for ann in annotations:
        class_id = ann["class_id"]
        polygon = ann["polygon"]
        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        if polygon and len(polygon) >= 2 and class_id != 3:
            draw.polygon(polygon, fill=color + (128,))

    result = Image.alpha_composite(img, mask_layer)
    result = result.convert("RGB")  # Конвертация в RGB для сохранения в PNG
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    result.save(output_dir)
    logger.info(f"Изображение с масками сохранено: {output_dir}")

def main():
    os.chdir(os.path.dirname(IMAGE_PATH) or os.getcwd())

    if not os.path.exists(IMAGE_PATH):
        logger.error(f"Изображение {IMAGE_PATH} не найдено. Проверьте путь или права доступа.")
        raise FileNotFoundError(f"Изображение {IMAGE_PATH} не найдено. "
                                f"Проверьте путь или права доступа.")

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Модель {MODEL_PATH} не найдена.")
        raise FileNotFoundError(f"Модель {MODEL_PATH} не найдена.")

    annotations = get_yolo_predictions(IMAGE_PATH, MODEL_PATH)
    if annotations is None:
        raise ValueError("Список с предсказаниями модели пуст.")

    draw_masks(IMAGE_PATH, annotations, OUTPUT_PATH)

if __name__ == "__main__":
    main()