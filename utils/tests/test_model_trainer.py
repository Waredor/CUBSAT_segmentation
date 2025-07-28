import logging
import logging.handlers
import unittest
import os
import json

from utils.model_trainer import ModelTrainer
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

        with open(self.valid_hyperparameters, mode='r', encoding='utf-8') as f:
            hyperparameters_data = json.load(f)

        hyperparameters_data['data_path'] = os.path.join(
            self.temp_dir,
            "valid_dataset.yaml"
        )
        self.hyperparameters = hyperparameters_data

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()

    def test_freeze_layers_success(self):
        """
        Тест test_freeze_layers_success проверяет работоспособность метода
        ModelTrainer.freeze_layers() с валидными входными данными
        """
        model_trainer = ModelTrainer(
            model_cfg=self.model_cfg,
            hyperparameters=self.hyperparameters,
            logger=self.logger,
        )
        model_trainer.freeze_layers(
            num_layers_to_freeze=self.hyperparameters["freeze_layers"]
        )
        layer_count = 0
        for param in model_trainer.model.model.parameters():
            if layer_count < self.hyperparameters["freeze_layers"]:
                self.assertFalse(param.requires_grad)
            layer_count += 1

        log_count = len(self.log_handler.buffer)
        self.assertEqual(log_count, 1)

    def test_train_model_success(self):
        """
        Тест test_train_model_success проверяет правильность работы метода
        ModelTrainer.train_model() при валидных входных данных
        """
        model_trainer = ModelTrainer(
            model_cfg=self.model_cfg,
            hyperparameters=self.hyperparameters,
            logger=self.logger
        )
        model = model_trainer.train_model()
        layer_count = 0
        for param in model_trainer.model.model.parameters():
            if layer_count < self.hyperparameters["freeze_layers"]:
                self.assertFalse(param.requires_grad)
            layer_count += 1

        self.assertIsInstance(model, YOLO)
        self.assertEqual(self.log_handler.buffer[0].getMessage(), "Starting training")
        self.assertEqual(self.log_handler.buffer[-1].getMessage(), "Training completed")