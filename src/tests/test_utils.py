import logging
import unittest
import os
import yaml
import json
import numpy as np

from unittest.mock import patch, MagicMock
from ultralytics.models.yolo.model import YOLO
from src.utils import ConfigManager, ModelTrainer, AnnotationProcessor


def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    return os.path.abspath(os.path.join(current_dir, '..', '..'))

project_root_path = get_project_root()

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(
            project_root_path,
            'src\\tests\\test_data'
        )
        self.model_cfg = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\model.pt'
        )
        self.valid_dataset = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\valid_dataset.yaml'
        )
        self.valid_hyperparameters = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\valid_hyperparameters.json'
        )
        with open(self.valid_dataset, mode='r', encoding='utf-8') as f:
            dataset_yaml_data = yaml.safe_load(f)

        dataset_yaml_data['path'] = self.temp_dir
        with open(self.valid_dataset, mode='w', encoding='utf-8') as f:
            yaml.safe_dump(dataset_yaml_data, f, encoding='utf-8')

        self.valid_hyperparameters = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\valid_hyperparameters.json'
        )
        self.invalid_dataset_key_error = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\invalid_dataset_key_error.yaml'
        )
        self.invalid_dataset_key_none = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\invalid_dataset_key_none.yaml'
        )
        self.invalid_dataset_key_not_a_directory = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\invalid_dataset_key_not_a_directory.yaml'
        )
        self.invalid_hyperparameters = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\invalid_hyperparameters.json'
        )
        self.invalid_json_to_parse = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\invalid_json_to_parse.json'
        )
        self.invalid_yaml_to_parse = os.path.join(
            project_root_path,
            'src\\tests\\test_data\\invalid_yaml_to_parse.yaml'
        )

    @patch('src.utils.logging.getLogger')
    def test_validate_config_success(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        config_manager.validate_config()
        self.assertEqual(mock_logger.info.call_count, 2)
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertEqual(calls[0], "Starting validation")
        self.assertEqual(calls[1], "Validation completed")

    @patch('src.utils.logging.getLogger')
    def test_validate_config_none_data_cfg(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=None,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()
        self.assertEqual(
            first=str(cm.exception),
            second="Parameter data_cfg is None"
        )

    @patch('src.utils.logging.getLogger')
    def test_validate_config_invalid_filepath_data_cfg(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=os.path.join('path', 'false_path.yaml'),
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(FileNotFoundError) as cm:
            config_manager.validate_config()
        self.assertEqual(
            first=str(cm.exception),
            second=os.path.join("path", "false_path.yaml") + " is not a path to file"
        )

    @patch('src.utils.logging.getLogger')
    def test_validate_config_invalid_filetype_data_cfg(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        filepath = os.path.join(
            project_root_path,
            'src',
            'tests',
            'test_data',
            'valid_dataset.txt'
        )
        config_manager = ConfigManager(
            data_cfg=filepath,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()
        self.assertEqual(
            first=str(cm.exception),
            second=f"{filepath} has invalid file extension"
        )

    @patch('src.utils.logging.getLogger')
    def test_validate_config_incorrect_type_data_dir(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=5,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()
        self.assertEqual(
            first=str(cm.exception),
            second="Parameter data_dir has incorrect type "
            "(expected: <class 'str'>, got: <class 'int'>)"
        )

    @patch('src.utils.logging.getLogger')
    def test_load_config_success(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        metadata = {
            "epochs": [int],
            "imgsz": [int],
            "batch": [int],
            "lr0": [float],
            "patience": [int],
            "device": [str, int],
            "optimizer": [str],
            "freeze_layers": [int],
            "data_path": [str],
            "class_names": [list],
            "output_dir": [str]
        }
        metadata_len = len(metadata.keys())
        hyperparams = config_manager.load_config()
        self.assertEqual(mock_logger.info.call_count, 5)
        self.assertEqual(type(hyperparams), dict)
        self.assertEqual(len(hyperparams.keys()), metadata_len)
        self.assertIn("epochs", hyperparams.keys())
        self.assertIn("imgsz", hyperparams.keys())
        self.assertIn("batch", hyperparams.keys())
        self.assertIn("lr0", hyperparams.keys())
        self.assertIn("patience", hyperparams.keys())
        self.assertIn("device", hyperparams.keys())
        self.assertIn("optimizer", hyperparams.keys())
        self.assertIn("freeze_layers", hyperparams.keys())
        self.assertIn("data_path", hyperparams.keys())
        self.assertIn("class_names", hyperparams.keys())
        self.assertIn("output_dir", hyperparams.keys())

    @patch('src.utils.logging.getLogger')
    def test_load_config_dataset_key_error(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_error,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(KeyError):
            config_manager.load_config()

    @patch('src.utils.logging.getLogger')
    def test_load_config_dataset_key_not_a_directory(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_not_a_directory,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(FileNotFoundError):
            config_manager.load_config()

    @patch('src.utils.logging.getLogger')
    def test_load_config_dataset_key_none(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_none,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()

    @patch('src.utils.logging.getLogger')
    def test_load_config_hyperparameters_key_error(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.invalid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(TypeError):
            config_manager.load_config()

    @patch('src.utils.logging.getLogger')
    def test_load_config_hyperparameters_json_error(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.invalid_json_to_parse,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()

    @patch('src.utils.logging.getLogger')
    def test_load_config_hyperparameters_yaml_error(self, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.invalid_yaml_to_parse,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(
            project_root_path,
            'src',
            'tests',
            'test_data'
        )
        self.model_cfg = os.path.join(
            project_root_path,
            'src',
            'tests',
            'test_data',
            'model.pt'
        )
        self.valid_dataset = os.path.join(
            project_root_path,
            'src',
            'tests',
            'test_data',
            'valid_dataset.yaml'
        )
        self.valid_hyperparameters = os.path.join(
            project_root_path,
            'src',
            'tests',
            'test_data',
            'valid_hyperparameters.json'
        )

    @patch('src.utils.logging.getLogger')
    def test_freeze_layers_success(self, mock_get_logger):
        """
        Тест test_freeze_layers_success проверяет работоспособность метода
        ModelTrainer.freeze_layers() с валидными входными данными
        """
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        hyperparameters = config_manager.load_config()
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        model_trainer = ModelTrainer(
            model_cfg=self.model_cfg,
            hyperparameters=hyperparameters
        )
        model_trainer.freeze_layers(
            num_layers_to_freeze=hyperparameters["freeze_layers"]
        )
        layer_count = 0
        for param in model_trainer.model.model.parameters():
            if layer_count < hyperparameters["freeze_layers"]:
                self.assertFalse(param.requires_grad)
            layer_count += 1
        self.assertEqual(mock_logger.info.call_count, 1)

    @patch('src.utils.logging.getLogger')
    def test_train_model_success(self, mock_get_logger):
        """
        Тест test_train_model_success проверяет правильность работы метода
        ModelTrainer.train_model() при валидных входных данных
        """
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir
        )
        hyperparameters = config_manager.load_config()

        mock_logger = MagicMock()
        mock_logger.level = logging.INFO
        mock_get_logger.return_value = mock_logger
        model_trainer = ModelTrainer(
            model_cfg=self.model_cfg,
            hyperparameters=hyperparameters
        )
        model = model_trainer.train_model()
        layer_count = 0
        for param in model_trainer.model.model.parameters():
            if layer_count < hyperparameters["freeze_layers"]:
                self.assertFalse(param.requires_grad)
            layer_count += 1

        self.assertIsInstance(model, YOLO)
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertEqual(calls[0], "Starting training")
        self.assertEqual(calls[-1], "Training completed")


class TestAnnotationProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.join(
            project_root_path,
            'src\\tests\\test_data'
        )

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
            output_dir=self.temp_dir
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
            output_dir=self.temp_dir
        )
        polygons = annotation_processor.mask_to_polygons(masks)
        self.assertEqual(type(polygons), list)
        self.assertEqual(len(polygons), 0)

    @patch('src.utils.logging.getLogger')
    def test_create_labelme_json_success(self, mock_get_logger):
        image_path = self.temp_dir + "\\images\\val\\0026.jpg"
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

        mock_logger = MagicMock()
        mock_logger.level = logging.INFO
        mock_get_logger.return_value = mock_logger
        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            output_dir=self.temp_dir
        )
        output_path = annotation_processor.create_labelme_json(
            image_path=image_path,
            masks=masks,
            labels=labels,
            class_names=class_names,
            output_dir=annotation_processor.output_dir
        )
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        file_dir = self.temp_dir + '\\0026.json'
        self.assertTrue(os.path.exists(file_dir))
        self.assertEqual(type(output_path), str)
        self.assertEqual(calls[-1], f"Created JSON-file: {file_dir}")
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

    @patch('src.utils.setup_logger')
    def test_create_labelme_json_error_empty_masks(self, mock_setup_logger):
        mock_setup_logger.return_value = logging.getLogger('test_logger')
        image_path = self.temp_dir + "\\images\\val\\0026.jpg"
        class_names = ['FT']

        masks = np.array([])

        num_objects = np.random.randint(1, 11)
        class_index = 0

        labels = np.full(num_objects, class_index, dtype=np.int64)

        annotation_processor = AnnotationProcessor(
            class_names=class_names,
            output_dir=self.temp_dir
        )

        with self.assertRaises(ValueError):
            annotation_processor.create_labelme_json(
                image_path=image_path,
                masks=masks,
                labels=labels,
                class_names=class_names,
                output_dir=annotation_processor.output_dir
            )

    @patch('src.utils.setup_logger')
    def test_create_labelme_json_error_empty_labels(self, mock_setup_logger):
        mock_setup_logger.return_value = logging.getLogger('test_logger')
        image_path = self.temp_dir + "\\images\\val\\0026.jpg"
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
            output_dir=self.temp_dir
        )

        with self.assertRaises(ValueError):
            annotation_processor.create_labelme_json(
                image_path=image_path,
                masks=masks,
                labels=labels,
                class_names=class_names,
                output_dir=annotation_processor.output_dir
            )

    @patch('src.utils.setup_logger')
    def test_create_labelme_json_error_incorrect_output_dir(self, mock_setup_logger):
        mock_setup_logger.return_value = logging.getLogger('test_logger')
        image_path = self.temp_dir + "\\images\\val\\0026.jpg"
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
            output_dir="\\wrong_dir\\no_dir"
        )

        with self.assertRaises(NotADirectoryError):
            annotation_processor.create_labelme_json(
                image_path=image_path,
                masks=masks,
                labels=labels,
                class_names=class_names,
                output_dir=annotation_processor.output_dir
            )

    @patch('src.utils.setup_logger')
    def test_create_labelme_json_error_incorrect_image_path(self, mock_setup_logger):
        mock_setup_logger.return_value = logging.getLogger('test_logger')
        wrong_image_path = self.temp_dir + "\\images\\val\\0022.jpg"
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
            output_dir=self.temp_dir
        )

        with self.assertRaises(FileNotFoundError):
            annotation_processor.create_labelme_json(
                image_path=wrong_image_path,
                masks=masks,
                labels=labels,
                class_names=class_names,
                output_dir=annotation_processor.output_dir
            )