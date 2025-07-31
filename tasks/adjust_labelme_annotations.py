import os
import json
import shutil
import cv2
import shapely
import numpy as np

from pathlib import Path
from shapely.geometry import Polygon


def load_image(image_path: str) -> np.ndarray:
    """
    Метод load_image() загружает изображение в оттенках серого
    Parameters:
        image_path (str): путь к файлу изображения
    Returns:
        img (np.ndarray): загруженное изображение
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Can't load image {image_path}")

    return img

def get_visible_contours(image: np.ndarray, threshold=30) -> list:
    """
    Метод get_visible_contours() находит контуры видимой части объекта на изображении
    Parameters:
        image (np.ndarray): numpy массив изображения
        threshold (int): порог для бинаризации (от 0 до 255)
    Returns:
        polygons(list): список полигонов
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            contour = contour.squeeze()
            if contour.ndim == 2:
                polygon = contour.tolist()
                polygons.append(polygon)

    return polygons

def is_polygon_visible(polygon: list, image: np.ndarray, threshold=30) -> bool:
    """
    Метод is_polygon_visible() проверяет, виден ли полигон на изображении.
    Возвращает True, если в области полигона есть пиксели выше порога
    Parameters:
        polygon (list): список с точками границ полигона
        image (np.ndarray): numpy массив изображения
        threshold (int): порог для бинаризации (от 0 до 255)
    """
    poly_points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [poly_points], 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return np.any(masked_image > threshold)

def adjust_polygon_to_visible(polygon: list, visible_polygons: list) -> None | list:
    """
    Метод adjust_polygon_to_visible() корректирует полигон,
    чтобы он соответствовал видимой части объекта
    Если полигон пересекается с видимыми контурами, возвращает новый полигон
    Parameters:
        polygon (list): список с точками границ полигона
        visible_polygons (list): список с видимыми полигонами
    Returns:
        None | list
    """
    if not visible_polygons:
        return None

    poly = Polygon(polygon)
    intersections = []

    for vis_poly in visible_polygons:
        vis_poly_shapely = Polygon(vis_poly)
        if poly.intersects(vis_poly_shapely):
            intersection = poly.intersection(vis_poly_shapely)
            if intersection.is_valid and not intersection.is_empty:
                if intersection.geom_type == "Polygon":
                    intersections.append(intersection)

                elif intersection.geom_type == "MultiPolygon":
                    intersections.extend(intersection.geoms)

    if not intersections:
        return None

    if len(intersections) == 1:
        result_poly = intersections[0]

    else:
        result_poly = shapely.unary_union(intersections)

    if result_poly.geom_type == "Polygon":
        return list(result_poly.exterior.coords)[:-1]

    elif result_poly.geom_type == "MultiPolygon":
        largest_poly = max(result_poly.geoms, key=lambda x: x.area)
        return list(largest_poly.exterior.coords)[:-1]

    return None

def process_annotations(
    json_path: str, image_path: str, output_json_path, valid_classes=None, threshold=30
) -> None:
    """
    Метод process_annotations() обрабатывает JSON аннотацию,
    корректируя полигоны для указанных классов
    Parameters:
        json_path (str): путь до исходного json afqkf
        image_path (str): путь до изображения
        output_json_path (str): путь до преобразованного json файла
        valid_classes (list): список допустимых классов
        threshold (int): порог для бинаризации (от 0 до 255)
    """
    if valid_classes is None:
        valid_classes = ["FT", "Engine", "Solar Panel"]

    with open(json_path, "r") as f:
        data = json.load(f)

    image = load_image(image_path)

    visible_polygons = get_visible_contours(image, threshold)

    new_shapes = []
    for shape in data.get("shapes", []):
        if shape["shape_type"] == "polygon" and shape.get("label") in valid_classes:
            polygon = shape["points"]

            if is_polygon_visible(polygon, image, threshold):
                adjusted_polygon = adjust_polygon_to_visible(polygon, visible_polygons)
                if adjusted_polygon:
                    shape["points"] = adjusted_polygon
                    new_shapes.append(shape)

        else:
            new_shapes.append(shape)

    data["shapes"] = new_shapes

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)

def main(
    input_json_dir: str, input_image_dir: str, output_json_dir: str, valid_classes=None, threshold=30
) -> None:
    """
    Метод main() обрабатывает все JSON файлы в указанной директории
    Parameters:
        input_json_dir (str): <UNK> <UNK> <UNK> json afqkf

    """
    if valid_classes is None:
        valid_classes = ["FT", "Engine", "Solar Panel"]

    os.makedirs(output_json_dir, exist_ok=True)

    for json_file in Path(input_json_dir).glob("*.json"):
        json_path = str(json_file)
        image_file = json_file.stem + ".png"
        image_path = str(Path(input_image_dir) / image_file)
        output_json_path = os.path.join(output_json_dir, json_file.name)

        if not os.path.exists(image_path):
            print(f"Изображение не найдено {image_path}")
            continue

        try:
            process_annotations(json_path, image_path, output_json_path, valid_classes, threshold)
            image_destination_path = os.path.join(output_json_dir, image_file)
            shutil.copy2(image_path, image_destination_path)

        except Exception as e:
            raise e

if __name__ == "__main__":
    input_json_dir = "C:\Python projects\Datasets\YOLOv11_CUBSAT_seg\CubSat\input"
    output_json_dir = "C:\Python projects\Datasets\YOLOv11_CUBSAT_seg\CubSat\output"
    threshold = 60
    valid_classes = ["FT", "Engine", "Solar Panel"]

    main(
        input_json_dir, input_json_dir, output_json_dir, valid_classes=valid_classes, threshold=threshold
    )

