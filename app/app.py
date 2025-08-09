import os
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from flask import Flask, request, render_template
import io
import base64

app = Flask(__name__)


init_path = os.path.abspath(__file__)


def get_project_root(start_path):
    current = start_path
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Корень проекта не найден")


project_root_path = get_project_root(init_path)


MODEL_PATH = os.path.join(project_root_path, 'app', 'models', 'yolo11n-seg_prod.pt')


LOG_DIR = os.path.join(project_root_path, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)  # Создаем папку logs, если она не существует


logging.basicConfig(
    format='%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s',
    level=logging.INFO,
    filename=os.path.join(LOG_DIR, 'inference_log.txt'),
    filemode='w',
    encoding='utf-8'
)
stream_handler = logging.StreamHandler()
rotating_file_handler = RotatingFileHandler(
    filename=os.path.join(LOG_DIR, 'inference_log.txt'),
    maxBytes=1048576,
    backupCount=3
)
stream_handler.setLevel(logging.INFO)
rotating_file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s')
stream_handler.setFormatter(formatter)
rotating_file_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addHandler(rotating_file_handler)


CLASS_NAMES = {0: "FT", 1: "Engine", 2: "Solar Panel"}
CLASS_COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}


def get_yolo_predictions(image_stream: io.BytesIO, model_path: str) -> list:
    """
    Получает предсказания от модели YOLOv11 из потока изображения.
    """
    if not os.path.exists(model_path):
        logger.warning(f"Модель {model_path} не найдена.")
        return []
    try:
        image_stream.seek(0)
        img = Image.open(image_stream)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        model = YOLO(model_path)
        results = model.predict(img, conf=0.5)
        annotations = []
        for result in results:
            if result.masks is not None:
                for mask, cls, box in zip(result.masks.xy, result.boxes.cls, result.boxes.xyxy):
                    class_id = int(cls)
                    polygon = [(float(x), float(y)) for x, y in mask]
                    x_min, y_min, _, _ = box
                    annotations.append({
                        "class_id": class_id,
                        "polygon": polygon,
                        "label_position": (float(x_min), float(y_min))
                    })
        return annotations
    except Exception as e:
        logger.error(f"Ошибка при выполнении инференса: {e}")
        return []


def draw_masks(image_stream: io.BytesIO, annotations: list) -> str | None:
    """
    Отрисовывает маски и подписи классов на изображении, возвращает изображение в base64.
    """
    try:
        image_stream.seek(0)
        img = Image.open(image_stream).convert("RGBA")
    except Exception as e:
        logger.error(f"Ошибка при открытии изображения: {e}")
        return None

    mask_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask_layer)


    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    if not annotations:
        logger.warning("Нет предсказанных масок для изображения.")
    else:
        for ann in annotations:
            class_id = ann["class_id"]
            polygon = ann["polygon"]
            label_position = ann["label_position"]
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(class_id, "Unknown")

            if polygon and len(polygon) >= 2:
                draw.polygon(polygon, fill=color + (128,))
                draw.text(label_position, class_name, fill=color + (255,), font=font)

    result = Image.alpha_composite(img, mask_layer)
    result = result.convert("RGB")


    buffered = io.BytesIO()
    result.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Файл не загружен")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="Файл не выбран")

        if file:
            image_stream = io.BytesIO()
            file.save(image_stream)

            annotations = get_yolo_predictions(image_stream, MODEL_PATH)
            if not annotations:
                return render_template('index.html', error="Не удалось получить предсказания")

            img_base64 = draw_masks(image_stream, annotations)
            if img_base64:
                return render_template('index.html', image=img_base64)
            else:
                return render_template('index.html', error="Ошибка при обработке изображения")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)