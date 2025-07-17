import json
import os
import logging
from logging.handlers import RotatingFileHandler
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


logging.basicConfig(
    format='%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s',
    level=logging.INFO,
    filename='background_adding_log.txt',
    filemode='w',
    encoding='utf-8'
)

stream_handler = logging.StreamHandler()
rotating_file_handler = RotatingFileHandler(
    filename='cubsat_log.txt',
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

def add_background_polygon(json_path: str,
                           output_path=None) -> None:
    """
    Метод add_background_polygon() отвечает за создание
    полигона фона для одного .json файла
    Parameters:
        json_path (str): путь к .json файлу с аннотациями
        output_path: путь к выходному .json файлу
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    image_polygon = Polygon([
        (0, 0),
        (img_width, 0),
        (img_width, img_height),
        (0, img_height)
    ])

    object_polygons = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            points = shape['points']
            if points[0] != points[-1]:
                points.append(points[0])
            try:
                poly = Polygon(points)
                if poly.is_valid:
                    object_polygons.append(poly)
            except Exception as e:
                logger.error(f"Ошибка при обработке полигона"
                             f"в {json_path}: {e}")

    if object_polygons:
        combined_objects = unary_union(object_polygons)
    else:
        combined_objects = Polygon()

    background_polygon = image_polygon.difference(combined_objects)
    background_shapes = []
    if isinstance(background_polygon, Polygon):
        points = list(background_polygon.exterior.coords)[:-1]
        background_shapes.append({
            'label': 'background',
            'points': points,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {}
        })

    elif isinstance(background_polygon, MultiPolygon):
        for poly in background_polygon.geoms:
            points = list(poly.exterior.coords)[:-1]
            background_shapes.append({
                'label': 'background',
                'points': points,
                'group_id': None,
                'shape_type': 'polygon',
                'flags': {}
            })

    data['shapes'].extend(background_shapes)

    if output_path is None:
        output_path = json_path
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Обработан файл: {json_path}")

def process_directory(input_dir: str, output_dir=None) -> None:
    """
    Метод process_directory() осуществляет создание аннотаций на основе
    исходных, лежащих в указанной директории
    Parameters:
        input_dir (str): путь к директории с .json файлами аннотаций
        output_dir: путь к выходной директории для сохранения .json файлов
    """
    logger.info(f"Начало работы в директории {input_dir}")
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename) if output_dir else None
            add_background_polygon(input_path, output_path)

    logger.info(f"Работа в директории {input_dir} завершена")

if __name__ == '__main__':
    annotations_dir = 'C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/'
    process_directory(input_dir=annotations_dir)