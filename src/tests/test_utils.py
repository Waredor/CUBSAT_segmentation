import logging
import unittest
import os

import torch.nn
import ultralytics
import yaml

from io import StringIO
from unittest.mock import patch, MagicMock
from src.utils import ConfigManager, ModelTrainer

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
while not os.path.exists(os.path.join(current_dir, '.venv')):
    current_dir = os.path.dirname(current_dir)
    if current_dir == os.path.dirname(current_dir):
        raise FileNotFoundError("Папка .venv не найдена в проекте")

project_root_path = os.path.abspath(current_dir)



class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = str(project_root_path) + '\\src\\tests\\test_data\\'
        self.model_cfg = str(project_root_path) + '\\src\\tests\\test_data\\model.pt'
        self.valid_dataset = (str(project_root_path) +
                              '\\src\\tests\\test_data\\valid_dataset.yaml')
        with open (file=self.valid_dataset, mode='r', encoding='utf-8') as f:
            dataset_yaml_data = yaml.safe_load(f)

        dataset_yaml_data['path'] = self.temp_dir
        with open (file=self.valid_dataset, mode='w', encoding='utf-8') as f:
            yaml.safe_dump(dataset_yaml_data, f, encoding='utf-8')

        self.valid_hyperparameters = (str(project_root_path) +
                                      '\\src\\tests\\test_data\\valid_hyperparameters.json')
        self.invalid_dataset_key_error = (str(project_root_path) +
                                          '\\src\\tests\\test_data\\invalid_dataset_key_error.yaml')
        self.invalid_dataset_key_none = (str(project_root_path) +
                                         '\\src\\tests\\test_data\\invalid_dataset_key_none.yaml')
        self.invalid_dataset_key_not_a_directory = (str(project_root_path) +
                                                    '\\src\\tests\\test_data\\'
                                                    'invalid_dataset_key_not_a_directory.yaml')
        self.invalid_hyperparameters = (str(project_root_path) +
                                        '\\src\\tests\\test_data\\invalid_hyperparameters.json')
        self.invalid_json_to_parse = (str(project_root_path) +
                                      '\\src\\tests\\test_data\\invalid_json_to_parse.json')
        self.invalid_yaml_to_parse = (str(project_root_path) +
                                      '\\src\\tests\\test_data\\invalid_yaml_to_parse.yaml')

    @patch('logging.getLogger')
    def test_validate_config_success(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с валидными конфигурационными файлами и путями к директориям
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )
        config_manager.validate_config()
        self.assertEqual(mock_logger.info.call_count, 2)
        calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertEqual(calls[0], "Начало валидации")
        self.assertEqual(calls[1], "Валидация завершена")

    @patch('logging.getLogger')
    def test_validate_config_none_data_cfg(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с аргументом data_cfg=None
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=None,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )

        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()

        self.assertEqual(str(cm.exception), "Переменная data_cfg имеет тип данных None")

    @patch('logging.getLogger')
    def test_validate_config_invalid_filepath_data_cfg(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с неправильным путем в аргументе data_cfg
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg='path\\false_path.yaml',
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )

        with self.assertRaises(FileNotFoundError) as cm:
            config_manager.validate_config()

        self.assertEqual(str(cm.exception), "path\\false_path.yaml не является путем к файлу")

    @patch('logging.getLogger')
    def test_validate_config_invalid_filetype_data_cfg(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с неправильным расширением файла в аргументе data_cfg
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        filepath = str(project_root_path) + '\\src\\tests\\test_data\\valid_dataset.txt'
        config_manager = ConfigManager(
            data_cfg=filepath,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )

        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()

        self.assertEqual(str(cm.exception), f"{filepath} имеет неверное расширение файла")

    @patch('logging.getLogger')
    def test_validate_config_incorrect_type_data_dir(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с аргументом type(data_dir)=int
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=5,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )

        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()

        self.assertEqual(str(cm.exception), "Переменная data_dir имеет неправильный "
                                  "тип данных"
                                  " (expected: <class 'str'>, "
                                  "got: <class 'int'>)")

    @patch('logging.getLogger')
    def test_load_config_success(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        с валидными конфигурационными файлами и путями к директориям
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
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
            "class_names":[dict],
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

    @patch('logging.getLogger')
    def test_load_config_dataset_key_error(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        с отсутсвующим ключом в конфигурационном .yaml файле датасета
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_error,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )
        with self.assertRaises(KeyError):
            config_manager.load_config()

    @patch('logging.getLogger')
    def test_load_config_dataset_key_not_a_directory(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        со значением, получаемым по ключу директории, в конфигурационном .yaml файле датасета
        и не являющимся директорией
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_not_a_directory,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )
        with self.assertRaises(NotADirectoryError):
            config_manager.load_config()

    @patch('logging.getLogger')
    def test_load_config_dataset_key_none(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        со значением None, получаемым по ключу, в конфигурационном .yaml файле датасета
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_none,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()

    @patch('logging.getLogger')
    def test_load_config_hyperparameters_key_error(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        со значением неверного типа, получаемым по ключу, в конфигурационном .json файле модели
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.invalid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )
        with self.assertRaises(TypeError):
            config_manager.load_config()

    @patch('logging.getLogger')
    def test_load_config_hyperparameters_json_error(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        с ошибкой в структуре конфигурационного .json файла модели
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.invalid_json_to_parse,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()

    @patch('logging.getLogger')
    def test_load_config_hyperparameters_yaml_error(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        с ошибкой в структуре конфигурационного .yaml файла датасета
        """
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        config_manager = ConfigManager(
            data_cfg=self.invalid_yaml_to_parse,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()


class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = str(project_root_path) + '\\src\\tests\\test_data\\'
        self.model_cfg = str(project_root_path) + '\\src\\tests\\test_data\\model.pt'
        self.valid_dataset = (str(project_root_path) +
                              '\\src\\tests\\test_data\\valid_dataset.yaml')
        self.valid_hyperparameters = (str(project_root_path) +
                                      '\\src\\tests\\test_data\\valid_hyperparameters.json')

    @patch('logging.getLogger')
    def test_freeze_layers_success(self, mock_get_logger):
        """
        Этот тест проверяет работу метода ModelTrainer.freeze_layers()
        при валидных конфигурационных файлах и параметрах модели
        """
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
        )
        hyperparameters = config_manager.load_config()
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        model_trainer = ModelTrainer(
            model_cfg=self.model_cfg,
            hyperparameters=hyperparameters,
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

    def test_train_model_success(self):
        """
        Этот тест проверяет работу метода ModelTrainer.train_model()
        при валидных конфигурационных файлах и параметрах модели
        """
        logger = logging.getLogger('test_logger')
        logger.setLevel(logging.DEBUG)
        log_output = StringIO()
        handler = logging.StreamHandler(log_output)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        with patch('src.utils.logging.getLogger', return_value=logger):
            config_manager = ConfigManager(
                data_cfg=self.valid_dataset,
                model_hyperparameters=self.valid_hyperparameters,
                data_dir=self.temp_dir,
                model_cfg=self.model_cfg,
                output_dir=self.temp_dir,
            )
            hyperparameters = config_manager.load_config()

            model_trainer = ModelTrainer(
                model_cfg=self.model_cfg,
                hyperparameters=hyperparameters,
            )
            model_trainer.train_model()

        handler.flush()
        logger.removeHandler(handler)
        log_content = log_output.getvalue().splitlines()

        info_logs = [line.split(' - ')[1] for line in log_content
                     if line.startswith('INFO')
                     and "Загружен файл конфигурации" not in line
                     and "compile_threads set to 1 for win32" not in line
                     and "Начало валидации" not in line
                     and "Валидация завершена" not in line]

        self.assertEqual(len(info_logs), 4)
        self.assertEqual(info_logs[0], "Начало обучения")
        self.assertEqual(info_logs[3], "Обучение завершено")

        log_output.close()