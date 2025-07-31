import json
import os
from shapely.geometry import Polygon
from pathlib import Path


def load_json_file(file_path):
    """Загружает JSON файл"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data, file_path):
    """Сохраняет JSON файл"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_polygon_inside(polygon1_points, polygon2_points):
    """Проверяет, находится ли polygon1 полностью внутри polygon2"""
    try:
        poly1 = Polygon(polygon1_points)
        poly2 = Polygon(polygon2_points)
        return poly1.within(poly2)
    except:
        return False


def process_labelme_file(file_path):
    """Обрабатывает один LabelMe JSON файл"""
    # Загружаем JSON
    data = load_json_file(file_path)

    # Получаем все shapes
    shapes = data.get('shapes', [])
    if not shapes:
        return False  # Нет объектов для обработки

    # Группируем shapes по label
    label_groups = {}
    for i, shape in enumerate(shapes):
        if shape['shape_type'] == 'polygon':
            label = shape['label']
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append((i, shape))

    # Список индексов для удаления
    shapes_to_remove = set()

    # Проверяем каждую группу label
    for label, shape_list in label_groups.items():
        # Сравниваем все пары полигонов в группе
        for i, (idx1, shape1) in enumerate(shape_list):
            for j, (idx2, shape2) in enumerate(shape_list):
                if i >= j:  # Пропускаем сравнение полигона с самим собой и повторные пары
                    continue

                points1 = shape1['points']
                points2 = shape2['points']

                # Проверяем, находится ли один полигон внутри другого
                if is_polygon_inside(points1, points2):
                    shapes_to_remove.add(idx1)
                elif is_polygon_inside(points2, points1):
                    shapes_to_remove.add(idx2)

    # Создаем новый список shapes, исключая удаляемые
    new_shapes = [shape for i, shape in enumerate(shapes) if i not in shapes_to_remove]

    # Если были изменения, обновляем JSON
    if len(new_shapes) < len(shapes):
        data['shapes'] = new_shapes
        save_json_file(data, file_path)
        return True
    return False


def process_directory(directory_path):
    """Обрабатывает все JSON файлы в указанной директории"""
    directory = Path(directory_path)
    processed_files = 0
    modified_files = 0

    # Проходим по всем JSON файлам в директории
    for file_path in directory.glob('*.json'):
        print(f"Обработка файла: {file_path}")
        try:
            if process_labelme_file(file_path):
                modified_files += 1
                print(f"Файл изменен: {file_path}")
            processed_files += 1
        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {str(e)}")

    print(f"\nОбработано файлов: {processed_files}")
    print(f"Изменено файлов: {modified_files}")


if __name__ == "__main__":

    directory_path = "C:\Python projects\Datasets\YOLOv11_CUBSAT_seg\CubSat"
    if not os.path.isdir(directory_path):
        print(f"Ошибка: {directory_path} не является директорией")

    process_directory(directory_path)