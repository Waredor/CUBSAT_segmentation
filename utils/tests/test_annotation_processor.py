import logging
import logging.handlers
import unittest
import os
import json
import numpy as np

from utils.annotation_processor import AnnotationProcessor

def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    return os.path.abspath(os.path.join(current_dir, '..', '..'))

project_root_path = get_project_root()

class TestAnnotationProcessor(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_config_manager_logger")
        self.logger.setLevel(logging.INFO)

        self.log_handler = logging.handlers.MemoryHandler(capacity=1000)
        self.logger.addHandler(self.log_handler)

        self.temp_dir = os.path.join(
            project_root_path,
            'utils',
            'tests',
            'test_data'
        )

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()

    def test_mask_to_polygons_success(self):
        """
        Тест test_mask_to_polygons_success проверяет работу метода
        AnnotationProcessor.mask_to_polygons() с корректными входными данными
        """
        height, width = 640, 640
        num_instances = 3

        masks = np.zeros((num_instances, height, width), dtype=np.uint8)

        for i in range(num_instances):
            center_x = np.random.randint(100, width - 100)
            center_y = np.random.randint(100, height - 100)
            axis_x = np.random.randint(50, 150)
            axis_y = np.random.randint(50, 150)

            y, x = np.ogrid[:height, :width]
            distance = ((x - center_x) / axis_x) ** 2 + ((y - center_y) / axis_y) ** 2
            masks[i][distance <= 1] = 1

        for i in range(1, num_instances):
            for j in range(i):
                overlap = masks[i] & masks[j]
                masks[i][overlap == 1] = 0

        class_names = ['FT']
        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            logger=self.logger
        )
        polygons = annotation_processor.mask_to_polygons(masks)
        self.assertEqual(type(polygons), list)
        self.assertNotEqual(len(polygons), 0)
        for polygon in polygons:
            self.assertTrue(len(polygon) >= 3)

    def test_mask_to_polygons_empty_masks(self):
        """
        Тест test_mask_to_polygons_empty_masks проверяет работу метода
        AnnotationProcessor.mask_to_polygons() с пустыми масками на входе
        """
        masks = np.array([])
        class_names = ['FT']
        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            logger=self.logger
        )
        polygons = annotation_processor.mask_to_polygons(masks)
        self.assertEqual(type(polygons), list)
        self.assertEqual(len(polygons), 0)

    def test_create_labelme_json_success(self):
        image_path = os.path.join(
            self.temp_dir,
            "images",
            "val",
            "0026.jpg"
        )
        class_names = ['FT']
        height, width = 640, 640
        num_instances = 3

        masks = np.zeros((num_instances, height, width), dtype=np.uint8)

        for i in range(num_instances):
            center_x = np.random.randint(100, width - 100)
            center_y = np.random.randint(100, height - 100)
            axis_x = np.random.randint(50, 150)
            axis_y = np.random.randint(50, 150)

            y, x = np.ogrid[:height, :width]
            distance = ((x - center_x) / axis_x) ** 2 + ((y - center_y) / axis_y) ** 2
            masks[i][distance <= 1] = 1

        for i in range(1, num_instances):
            for j in range(i):
                overlap = masks[i] & masks[j]
                masks[i][overlap == 1] = 0

        num_objects = np.random.randint(1, 11)
        class_index = 0

        labels = np.full(num_objects, class_index, dtype=np.int64)

        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            logger=self.logger
        )
        output_path = annotation_processor.create_labelme_json(
            image_path=image_path,
            masks=masks,
            labels=labels,
            output_dir=self.temp_dir
        )

        file_dir = os.path.join(self.temp_dir, "0026.json")

        self.assertTrue(os.path.exists(file_dir))
        self.assertEqual(type(output_path), str)
        self.assertEqual(
            self.log_handler.buffer[-1].getMessage(),
            f"Created JSON-file: {file_dir}"
        )

        with open(file_dir, mode="r", encoding='utf-8') as f:
            json_annotations = json.load(f)

        for key, value in json_annotations.items():
            if key == "version" or key == "imagePath" or key == "imageData":
                self.assertEqual(type(value), str)

            elif key == "flags":
                self.assertEqual(type(value), dict)

            elif key == "imageHeight" or key == "imageWidth":
                self.assertEqual(type(value), int)

            elif key == "shapes":
                self.assertEqual(type(value), list)

        for shape in json_annotations["shapes"]:
            self.assertEqual(type(shape), dict)

            for key, value in shape.items():
                if key == "label" or key == "shape_type":
                    self.assertEqual(type(value), str)

                elif key == "points":
                    self.assertEqual(type(value), list)

                elif key == "flags":
                    self.assertEqual(type(value), dict)

    def test_create_labelme_json_error_empty_masks(self):
        image_path = os.path.join(
            self.temp_dir,
            "images",
            "val",
            "0026.jpg"
        )
        class_names = ['FT']

        masks = np.array([])

        num_objects = np.random.randint(1, 11)
        class_index = 0

        labels = np.full(num_objects, class_index, dtype=np.int64)

        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            logger=self.logger
        )

        with self.assertRaises(ValueError):
            annotation_processor.create_labelme_json(
                image_path=image_path,
                masks=masks,
                labels=labels,
                output_dir=self.temp_dir
            )

    def test_create_labelme_json_error_empty_labels(self):
        image_path = os.path.join(
            self.temp_dir,
            "images",
            "val",
            "0026.jpg"
        )
        class_names = ['FT']
        height, width = 640, 640
        num_instances = 3

        masks = np.zeros((num_instances, height, width), dtype=np.uint8)

        for i in range(num_instances):
            center_x = np.random.randint(100, width - 100)
            center_y = np.random.randint(100, height - 100)
            axis_x = np.random.randint(50, 150)
            axis_y = np.random.randint(50, 150)

            y, x = np.ogrid[:height, :width]
            distance = ((x - center_x) / axis_x) ** 2 + ((y - center_y) / axis_y) ** 2
            masks[i][distance <= 1] = 1

        for i in range(1, num_instances):
            for j in range(i):
                overlap = masks[i] & masks[j]
                masks[i][overlap == 1] = 0

        labels = np.array([])

        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            logger=self.logger
        )

        with self.assertRaises(ValueError):
            annotation_processor.create_labelme_json(
                image_path=image_path,
                masks=masks,
                labels=labels,
                output_dir=self.temp_dir
            )

    def test_create_labelme_json_error_incorrect_output_dir(self):
        image_path = os.path.join(
            self.temp_dir,
            "images",
            "val",
            "0026.jpg"
        )
        class_names = ['FT']
        height, width = 640, 640
        num_instances = 3

        masks = np.zeros((num_instances, height, width), dtype=np.uint8)

        for i in range(num_instances):
            center_x = np.random.randint(100, width - 100)
            center_y = np.random.randint(100, height - 100)
            axis_x = np.random.randint(50, 150)
            axis_y = np.random.randint(50, 150)

            y, x = np.ogrid[:height, :width]
            distance = ((x - center_x) / axis_x) ** 2 + ((y - center_y) / axis_y) ** 2
            masks[i][distance <= 1] = 1

        for i in range(1, num_instances):
            for j in range(i):
                overlap = masks[i] & masks[j]
                masks[i][overlap == 1] = 0

        num_objects = np.random.randint(1, 11)
        class_index = 0

        labels = np.full(num_objects, class_index, dtype=np.int64)

        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            logger=self.logger
        )

        with self.assertRaises(NotADirectoryError):
            annotation_processor.create_labelme_json(
                image_path=image_path,
                masks=masks,
                labels=labels,
                output_dir="\\wrong_dir\\no_dir"
            )

    def test_create_labelme_json_error_incorrect_image_path(self):
        wrong_image_path = os.path.join(
            self.temp_dir,
            "images",
            "val",
            "0022.jpg"
        )
        class_names = ['FT']
        height, width = 640, 640
        num_instances = 3

        masks = np.zeros((num_instances, height, width), dtype=np.uint8)

        for i in range(num_instances):
            center_x = np.random.randint(100, width - 100)
            center_y = np.random.randint(100, height - 100)
            axis_x = np.random.randint(50, 150)
            axis_y = np.random.randint(50, 150)

            y, x = np.ogrid[:height, :width]
            distance = ((x - center_x) / axis_x) ** 2 + ((y - center_y) / axis_y) ** 2
            masks[i][distance <= 1] = 1

        for i in range(1, num_instances):
            for j in range(i):
                overlap = masks[i] & masks[j]
                masks[i][overlap == 1] = 0

        num_objects = np.random.randint(1, 11)
        class_index = 0

        labels = np.full(num_objects, class_index, dtype=np.int64)

        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            logger=self.logger
        )

        with self.assertRaises(FileNotFoundError):
            annotation_processor.create_labelme_json(
                image_path=wrong_image_path,
                masks=masks,
                labels=labels,
                output_dir=self.temp_dir
            )