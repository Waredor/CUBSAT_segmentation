import logging
import logging.handlers
import unittest
import os
import json
import yaml

from utils.model_trainer import train_model
from ultralytics import YOLO

def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    return os.path.abspath(os.path.join(current_dir, '..', '..'))

project_root_path = get_project_root()

class TestModelTrainer(unittest.TestCase):
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

        self.model_cfg = os.path.join(
            project_root_path,
            'utils',
            'tests',
            'test_data',
            'model.pt'
        )

        self.valid_hyperparameters = os.path.join(
            project_root_path,
            'utils',
            'tests',
            'test_data',
            'model_trainer_valid_hyperparameters.json'
        )

        self.valid_dataset = os.path.join(
            project_root_path,
            'utils',
            'tests',
            'test_data',
            'valid_config.yaml'
        )

        with open(self.valid_dataset, mode='r', encoding='utf-8') as f:
            dataset_yaml_data = yaml.safe_load(f)

        dataset_yaml_data['path'] = self.temp_dir
        with open(self.valid_dataset, mode='w', encoding='utf-8') as f:
            yaml.safe_dump(dataset_yaml_data, f, encoding='utf-8')

        self.valid_hyperparameters = os.path.join(
            project_root_path,
            'utils',
            'tests',
            'test_data',
            'valid_hyperparameters.json'
        )

        with open(self.valid_hyperparameters, mode='r', encoding='utf-8') as f:
            hyperparameters_data = json.load(f)

        hyperparameters_data['data_path'] = os.path.join(
            self.temp_dir,
            "valid_config.yaml"
        )
        self.hyperparameters = hyperparameters_data

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()


    def test_train_model_success(self):
        """
        Тест test_train_model_success проверяет правильность работы метода
        train_model() при валидных входных данных
        """
        model = YOLO(self.model_cfg)
        model = train_model(model=model, hyperparameters=self.hyperparameters, logger=self.logger, augment=False)
        layer_count = 0
        for param in model.model.parameters():
            if layer_count < self.hyperparameters["freeze_layers"]:
                self.assertFalse(param.requires_grad)
            layer_count += 1

        self.assertIsInstance(model, YOLO)
        self.assertEqual(self.log_handler.buffer[0].getMessage(), "Starting training")
        self.assertEqual(self.log_handler.buffer[-1].getMessage(), "Training completed")