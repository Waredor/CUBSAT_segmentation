import os
import logging
import glob
import base64
import json
import cv2
import numpy as np

from pathlib import Path
from PIL import Image

class AnnotationProcessor:
    """
    Класс AnnotationProcessor создает аннотации в формате LabelMe .json
    к тестовым изображениям.
    Parameters:
        class_names (list): список с именами используемых классов.
        logger (logging.Logger): объект логгера.
    """

    def __init__(self, class_names: list, logger: logging.Logger) -> None:
        self.class_names = class_names
        self.logger = logger

    def mask_to_polygons(self, mask: np.ndarray) -> list:
        """
        Метод mask_to_polygons преобразует маски объектов, полученные в результате инференса,
        в список полигонов.
        Parameters:
            mask (np.ndarray): numpy маски объектов, полученные в результате инференса.
        Returns:
            polygons (list): список полигонов объектов.
        """
        polygons = []
        if mask.ndim == 3:
            for i in range(mask.shape[0]):
                single_mask = mask[i].astype(np.uint8)
                contours, _ = cv2.findContours(
                    single_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    if len(contour) >= 3:
                        polygon = contour.squeeze().tolist()
                        polygons.append(polygon)
        else:
            single_mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                single_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if len(contour) >= 3:
                    polygon = contour.squeeze().tolist()
                    polygons.append(polygon)

        return polygons

    def create_labelme_json(
            self, image_path: str, masks: np.ndarray, labels: np.ndarray, output_dir: str
    ) -> str:
        """
        Метод create_labelme_json() осуществляет создание аннотаций к инференсу
        из InferenceRunner в формате .json
        Parameters:
            image_path (str): путь к изображению, для которого создается аннотация.
            masks (np.ndarray): numpy массив масок полигонов объектов на изображении.
            labels (np.ndarray): numpy массив меток классов полигонов на изображении.
            output_dir (str): выходная директория для сохранения .json аннотаций
                к изображениям.
        """
        if not os.path.exists(output_dir):
            self.logger.error(f"Incorrect directory {output_dir}")
            raise NotADirectoryError(f"Incorrect directory {output_dir}")

        if not os.path.isfile(image_path):
            self.logger.error(f"{image_path} is not a path to a file")
            raise FileNotFoundError(f"{image_path} is not a path to a file")

        if np.size(masks) == 0:
            self.logger.error(f"{masks} is an empty array")
            raise ValueError(f"{masks} is an empty array")

        if np.size(labels) == 0:
            self.logger.error(f"{labels} is an empty array")
            raise ValueError(f"{labels} is an empty array")

        image_name = Path(image_path).name
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        _, buffer = cv2.imencode(".jpg", image)
        image_data = base64.b64encode(buffer).decode("utf-8")

        labelme_data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_name,
            "imageData": image_data,
            "imageHeight": height,
            "imageWidth": width
        }

        for mask, label in zip(masks, labels):
            polygons = self.mask_to_polygons(mask.astype(np.uint8))
            for polygon in polygons:
                shape = {
                    "label": self.class_names[int(label)],
                    "points": polygon,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                labelme_data["shapes"].append(shape)

        output_path = os.path.join(
            output_dir,
            image_name.replace(".jpg", ".json").replace(".png", ".json")
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, mode="w", encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2)
        self.logger.info(f"Created JSON-file: {output_path}")
        return output_path

    def convert_labelme_to_yolo(self, labelme_annotations_path: str,
                                yolo_annotations_path: str) -> None:
        """
        Метод convert_labelme_to_yolo() конвертирует аннотации изображений в формате .json LabelMe
        в формат YOLOv11 .txt
        Parameters:
            labelme_annotations_path (str): путь к директории с аннотациями в формате .json LabelMe.
            yolo_annotations_path (str): путь к директории с аннотациями в формате .txt YOLOv11.
        Raises:
            NotADirectoryError: если путь не является директорией
            FileNotFoundError: если файл по искомому пути не найден
        """
        if not os.path.isdir(labelme_annotations_path):
            self.logger.error(f"{labelme_annotations_path} is not a directory")
            raise NotADirectoryError(f"{labelme_annotations_path} is not a directory")

        if not os.path.isdir(yolo_annotations_path):
            self.logger.error(f"{yolo_annotations_path} is not a directory")
            raise NotADirectoryError(f"{yolo_annotations_path} is not a directory")

        class_map = {name: idx for idx, name in enumerate(self.class_names)}

        os.makedirs(yolo_annotations_path, exist_ok=True)
        json_files = glob.glob(os.path.join(labelme_annotations_path, "*.json"))
        if len(json_files) == 0:
            self.logger.warning(f"There are no annotations in {labelme_annotations_path}")
            raise FileNotFoundError(f"There are no annotations in {labelme_annotations_path}")

        self.logger.info("Starting annotations convertation")
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            except json.decoder.JSONDecodeError:
                self.logger.warning(f"Empty file {f}")
                continue

            image_filename = data.get('imagePath')
            image_path = os.path.join(labelme_annotations_path, image_filename)

            if not os.path.exists(image_path):
                self.logger.warning(f"There are no images in: {image_path}")
                continue

            with Image.open(image_path) as img:
                w, h = img.size

            yolo_lines = []
            for shape in data['shapes']:
                label = shape.get('label')
                points = shape.get('points', [])

                if label not in class_map:
                    self.logger.warning(f"Skipped class: '{label}' в {json_path}")
                    continue

                if len(points) < 3:
                    self.logger.warning(f"Skipped object with a few points {json_path}")
                    continue

                class_id = class_map[label]

                norm_points = []
                for x, y in points:
                    norm_x = round(x / w, 6)
                    norm_y = round(y / h, 6)
                    norm_points.extend([norm_x, norm_y])

                yolo_line = f"{class_id} " + " ".join(map(str, norm_points))
                yolo_lines.append(yolo_line)

            base_name = os.path.splitext(os.path.basename(json_path))[0]
            txt_path = os.path.join(yolo_annotations_path, base_name + ".txt")

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))
                self.logger.info(f"Converted {base_name}.txt")

        self.logger.info("Annotations convertation is finished")