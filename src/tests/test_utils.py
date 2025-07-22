import unittest
import sys
import os
import yaml
import json
import cv2
import numpy as np

from unittest.mock import patch, MagicMock
from src.utils import ConfigManager


current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
while not os.path.exists(os.path.join(current_dir, '.venv')):
    current_dir = os.path.dirname(current_dir)
    if current_dir == os.path.dirname(current_dir):
        raise FileNotFoundError("Папка .venv не найдена в проекте")

project_root_path = os.path.abspath(current_dir)

VALID_DATASET = str(project_root_path) + '\\src\\tests\\test_data\\valid_dataset.yaml'
INVALID_DATASET = str(project_root_path) + '\\src\\tests\\test_data\\invalid_dataset.yaml'
VALID_HYPERPARAMETERS = str(project_root_path) + '\\src\\tests\\test_data\\valid_hyperparameters.json'
INVALID_HYPERPARAMETERS = str(project_root_path) + '\\src\\tests\\test_data\\invalid_hyperparameters.json'
TEMP_DIR = str(project_root_path) + '\\src\\tests\\test_data\\'
MODEL_CFG = str(project_root_path) + '\\src\\tests\\test_data\\model.pt'

class TestConfigManager(unittest.TestCase):
    @patch('logging.getLogger')
    def test_validate_config_success(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с валидными конфигурационными файлами и путями к директориям
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=VALID_DATASET,
            model_hyperparameters=VALID_HYPERPARAMETERS,
            data_dir=TEMP_DIR,
            model_cfg=MODEL_CFG,
            output_dir=TEMP_DIR,
        )
        config_manager.validate_config()
        self.assertEqual(mock_logger.info.call_count, 2)
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertEqual(calls[0], "Начало валидации")
        self.assertEqual(calls[1], "Валидация завершена")

