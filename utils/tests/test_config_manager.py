import logging
import logging.handlers
import unittest
import os
import yaml

from utils.config_manager import ConfigManager

def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    return os.path.abspath(os.path.join(current_dir, '..', '..'))

project_root_path = get_project_root()

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_config_manager_logger")
        self.logger.setLevel(logging.INFO)

        self.log_handler = logging.handlers.MemoryHandler(capacity=1000)
        self.logger.addHandler(self.log_handler)

        self.temp_dir = os.path.join(
            project_root_path,
            'utils\\tests\\test_data'
        )
        self.model_cfg = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\model.pt'
        )
        self.valid_dataset = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\valid_dataset.yaml'
        )
        self.valid_hyperparameters = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\valid_hyperparameters.json'
        )
        with open(self.valid_dataset, mode='r', encoding='utf-8') as f:
            dataset_yaml_data = yaml.safe_load(f)

        dataset_yaml_data['path'] = self.temp_dir
        with open(self.valid_dataset, mode='w', encoding='utf-8') as f:
            yaml.safe_dump(dataset_yaml_data, f, encoding='utf-8')

        self.valid_hyperparameters = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\valid_hyperparameters.json'
        )
        self.invalid_dataset_key_error = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_dataset_key_error.yaml'
        )
        self.invalid_dataset_key_none = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_dataset_key_none.yaml'
        )
        self.invalid_dataset_key_not_a_directory = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_dataset_key_not_a_directory.yaml'
        )
        self.invalid_hyperparameters = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_hyperparameters.json'
        )
        self.invalid_json_to_parse = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_json_to_parse.json'
        )
        self.invalid_yaml_to_parse = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_yaml_to_parse.yaml'
        )

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()

    def test_validate_config_success(self):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с валидными конфигурационными файлами
        """
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        config_manager.validate_config()

        log_count = len(self.log_handler.buffer)
        self.assertEqual(log_count, 2)

        self.assertEqual(self.log_handler.buffer[0].getMessage(), "Starting validation")
        self.assertEqual(self.log_handler.buffer[-1].getMessage(), "Validation completed")

    def test_validate_config_none_data_cfg(self):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с переданным параметром data_cfg=None
        """
        config_manager = ConfigManager(
            data_cfg=None,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()
        self.assertEqual(
            first=str(cm.exception),
            second="Parameter data_cfg is None"
        )

    def test_validate_config_invalid_filepath_data_cfg(self):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с неверным путем до data_cfg
        """
        config_manager = ConfigManager(
            data_cfg=os.path.join('path', 'false_path.yaml'),
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(FileNotFoundError) as cm:
            config_manager.validate_config()
        self.assertEqual(
            first=str(cm.exception),
            second=os.path.join("path", "false_path.yaml") + " is not a path to file"
        )

    def test_validate_config_invalid_filetype_data_cfg(self):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с неверным расширением файла data_cfg
        """
        filepath = os.path.join(
            project_root_path,
            'utils',
            'tests',
            'test_data',
            'valid_dataset.txt'
        )
        config_manager = ConfigManager(
            data_cfg=filepath,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()
        self.assertEqual(
            first=str(cm.exception),
            second=f"{filepath} has invalid file extension"
        )

    def test_validate_config_incorrect_type_data_dir(self):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        с неверным типом данных data_dir
        """
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=5,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(ValueError) as cm:
            config_manager.validate_config()
        self.assertEqual(
            first=str(cm.exception),
            second="Parameter data_dir has incorrect type "
            "(expected: <class 'str'>, got: <class 'int'>)"
        )

    def test_load_config_success(self):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        с валидными конфигурационными файлами
        """
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
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

        log_count = len(self.log_handler.buffer)

        self.assertEqual(log_count, 5)
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

    def test_load_config_dataset_key_error(self):
        """
        Этот тест проверяет работу метода ConfigManager.validate_config()
        на выброс ошибки KeyError при отсутствующем ключе в конфигурационном файле
        датасета
        """
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_error,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(KeyError):
            config_manager.load_config()

    def test_load_config_dataset_key_not_a_directory(self):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        со значением, получаемым по ключу из конфигурационного файла датасета,
        которое не является директорией
        """
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_not_a_directory,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(FileNotFoundError):
            config_manager.load_config()

    def test_load_config_dataset_key_none(self):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        со значением, получаемым по ключу из конфигурационного файла датасета,
        которое является None
        """
        config_manager = ConfigManager(
            data_cfg=self.invalid_dataset_key_none,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()

    def test_load_config_hyperparameters_key_error(self):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        с ошибкой KeyError, возникающей при валидации .json файла с гиперпараметрами
        модели.
        """
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.invalid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(TypeError):
            config_manager.load_config()

    def test_load_config_hyperparameters_json_error(self):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        с ошибкой JSONError, возникающей при парсинге некорректного .json файла.
        """
        config_manager = ConfigManager(
            data_cfg=self.valid_dataset,
            model_hyperparameters=self.invalid_json_to_parse,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()

    def test_load_config_hyperparameters_yaml_error(self):
        """
         Этот тест проверяет работу метода ConfigManager.load_config()
         с ошибкой YAMLError, возникающей при парсинге некорректного .yaml файла.
         """
        config_manager = ConfigManager(
            data_cfg=self.invalid_yaml_to_parse,
            model_hyperparameters=self.valid_hyperparameters,
            data_dir=self.temp_dir,
            model_cfg=self.model_cfg,
            output_dir=self.temp_dir,
            logger=self.logger
        )
        with self.assertRaises(ValueError):
            config_manager.load_config()