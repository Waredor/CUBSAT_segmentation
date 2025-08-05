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
        self.valid_config = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\valid_config.yaml'
        )
        self.invalid_config_extension = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\valid_config.txt'
        )
        self.invalid_yaml_to_parse = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_yaml_to_parse.yaml'
        )
        self.invalid_config_key_error_first_level = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_key_error_first_level.yaml'
        )
        self.invalid_config_key_error_paths = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_key_error_paths.yaml'
        )
        self.invalid_config_key_error_names = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_key_error_names.yaml'
        )
        self.invalid_config_key_error_extensions = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_key_error_extensions.yaml'
        )
        self.invalid_config_key_error_model_hyperparameters = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_key_error_model_hyperparameters.yaml'
        )
        self.invalid_config_key_error_dataset_cfg = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_key_error_dataset_cfg.yaml'
        )
        self.invalid_config_value_error_dataset_cfg = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_value_error_dataset_cfg.yaml'
        )
        self.invalid_config_file_not_found_error_dataset_cfg = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_file_not_found_error_dataset_cfg.yaml'
        )
        self.invalid_config_not_a_directory_error_dataset_cfg = os.path.join(
            project_root_path,
            'utils\\tests\\test_data\\invalid_config_not_a_directory_error_dataset_cfg.yaml'
        )

        with open(self.valid_config, mode='r', encoding='utf-8') as f:
            dataset_yaml_data = yaml.safe_load(f)

        dataset_yaml_data['dataset_cfg']['path'] = self.temp_dir
        dataset_yaml_data['paths']['inference_images_dir'] = os.path.join(
            self.temp_dir, 'images', 'test'
        )
        dataset_yaml_data['paths']['labelme_inference_annotations_dir'] = os.path.join(
            self.temp_dir, 'labels', 'test'
        )
        dataset_yaml_data['paths']['raw_data_dir'] = self.temp_dir
        with open(self.valid_config, mode='w', encoding='utf-8') as f:
            yaml.safe_dump(dataset_yaml_data, f, encoding='utf-8')

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()

    def test_load_yaml_success(self):
        """
        Этот тест проверяет работу метода ConfigManager.load_yaml()
        с валидным конфигурационным файлом.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        data = config_manager.load_yaml()
        self.assertEqual(type(data), dict)

    def test_load_yaml_failure(self):
        """
        Этот тест проверяет работу метода ConfigManager.load_yaml()
        с ошибкой в синтаксисе конфигурационного файла.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_yaml_to_parse,
            logger=self.logger
        )

        with self.assertRaises(ValueError):
            config_manager.load_yaml()

    def test_validate_param_failure_none_type(self):
        """
        Этот тест проверяет работу метода ConfigManager._validate_param()
        при передаче в качестве параметра NoneType.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        with self.assertRaises(ValueError):
            config_manager._validate_param(None, config_manager.metadata[0])

    def test_validate_param_failure_negative_int_value(self):
        """
        Этот тест проверяет работу метода ConfigManager._validate_param()
        при передаче в качестве параметра целого отрицательного числа.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        with self.assertRaises(ValueError):
            config_manager._validate_param(-1, config_manager.metadata[15])

    def test_validate_param_failure_negative_float_value(self):
        """
        Этот тест проверяет работу метода ConfigManager._validate_param()
        при передаче в качестве параметра дробного отрицательного числа.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        with self.assertRaises(ValueError):
            config_manager._validate_param(-0.55, config_manager.metadata[18])

    def test_validate_param_failure_type_error(self):
        """
        Этот тест проверяет работу метода ConfigManager._validate_param()
        при несоответсвии типа данных передаваемого параметра и требуемого типа данных.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        with self.assertRaises(TypeError):
            config_manager._validate_param(5, config_manager.metadata[0])

    def test_validate_param_failure_empty_string(self):
        """
        Этот тест проверяет работу метода ConfigManager._validate_param()
        при передачи в качестве параметра пустой строки.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        with self.assertRaises(ValueError):
            config_manager._validate_param('', config_manager.metadata[0])

    def test_validate_param_failure_file_not_found_error(self):
        """
        Этот тест проверяет работу метода ConfigManager._validate_param()
        при передачи в качестве параметра строки не являющейся путем к файлу,
        но имеющей флаг is_file в метаданных.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        with self.assertRaises(FileNotFoundError):
            config_manager._validate_param(
                'wrong/path/to/file.file', config_manager.metadata[1]
            )

    def test_validate_param_failure_not_a_directory_error(self):
        """
        Этот тест проверяет работу метода ConfigManager._validate_param()
        при передачи в качестве параметра строки не являющейся путем к директории,
        но имеющей флаг is_dir в метаданных.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        with self.assertRaises(NotADirectoryError):
            config_manager._validate_param(
                'wrong/path/to/dir', config_manager.metadata[0]
            )

    def test_validate_param_failure_wrong_extension(self):
        """
        Этот тест проверяет работу метода ConfigManager._validate_param()
        при передачи в качестве параметра пути к файлу с неверным расширением.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        with self.assertRaises(ValueError):
            config_manager._validate_param(
                self.invalid_config_extension, config_manager.metadata[1]
            )

    def test_check_input_success(self):
        """
        Этот тест проверяет работу метода ConfigManager.check_input()
        с валидными переменными.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        for el in config_manager.metadata[0:2]:
            self.assertTrue(el.name in config_manager.params.keys())

        config_manager.check_input()

        log_count = len(self.log_handler.buffer)
        self.assertEqual(log_count, 2)

        self.assertEqual(self.log_handler.buffer[0].getMessage(), "Starting checking input")
        self.assertEqual(self.log_handler.buffer[-1].getMessage(), "Checking input completed")

    def test_check_yaml_file_success(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с валидным конфигурационным файлом
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        data = config_manager.load_yaml()
        config_manager._check_yaml_file(data)

        log_count = len(self.log_handler.buffer)
        self.assertEqual(log_count, 2)

        self.assertEqual(
            self.log_handler.buffer[0].getMessage(), "Starting validation of YAML file"
        )
        self.assertEqual(
            self.log_handler.buffer[-1].getMessage(), "Validated successfully"
        )

    def test_check_yaml_file_failure_key_error_first_level(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с отсутствующим ключом первого уровня в словаре с данными из .yaml файла конфигурации
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_key_error_first_level,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(KeyError):
            config_manager._check_yaml_file(data)

    def test_check_yaml_file_failure_key_error_paths(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с отсутствующим ключом в словаре, получаемому по ключу 'paths' из .yaml
        файла конфигурации.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_key_error_paths,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(KeyError):
            config_manager._check_yaml_file(data)

    def test_check_yaml_file_failure_key_error_names(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с отсутствующим ключом в словаре, получаемому по ключу 'names' из .yaml
        файла конфигурации.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_key_error_names,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(KeyError):
            config_manager._check_yaml_file(data)

    def test_check_yaml_file_failure_key_error_extensions(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с отсутствующим ключом в словаре, получаемому по ключу 'extensions' из .yaml
        файла конфигурации.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_key_error_extensions,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(KeyError):
            config_manager._check_yaml_file(data)

    def test_check_yaml_file_failure_key_error_model_hyperparameters(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с отсутствующим ключом в словаре, получаемому по ключу 'model_hyperparameters' из .yaml
        файла конфигурации.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_key_error_model_hyperparameters,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(KeyError):
            config_manager._check_yaml_file(data)

    def test_check_yaml_file_failure_key_error_dataset_cfg(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с отсутствующим ключом в словаре, получаемому по ключу 'dataset_cfg' из .yaml
        файла конфигурации.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_key_error_dataset_cfg,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(KeyError):
            config_manager._check_yaml_file(data)

    def test_check_yaml_file_failure_value_error_dataset_cfg(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с несовпадающим количеством классов в списке class_names и указанным числом классов nc
        в словаре, получаемому по ключу 'dataset_cfg' из .yaml
        файла конфигурации.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_value_error_dataset_cfg,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(ValueError):
            config_manager._check_yaml_file(data)

    def test_check_yaml_file_failure_file_not_found_error_dataset_cfg(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с некорректным путем к директории train или val в словаре,
        получаемому по ключу 'dataset_cfg' из .yaml файла конфигурации.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_file_not_found_error_dataset_cfg,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(FileNotFoundError):
            config_manager._check_yaml_file(data)

    def test_check_yaml_file_failure_not_a_directory_error_dataset_cfg(self):
        """
        Этот тест проверяет работу метода ConfigManager._check_yaml_file()
        с путем, не являющийся путем к директории train или val в словаре,
        получаемому по ключу 'dataset_cfg' из .yaml файла конфигурации.
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.invalid_config_not_a_directory_error_dataset_cfg,
            logger=self.logger
        )
        data = config_manager.load_yaml()

        with self.assertRaises(NotADirectoryError):
            config_manager._check_yaml_file(data)

    def test_load_config_success(self):
        """
        Этот тест проверяет работу метода ConfigManager.load_config()
        при валидных переменных
        """
        config_manager = ConfigManager(
            project_root=self.temp_dir,
            pipeline_cfg=self.valid_config,
            logger=self.logger
        )
        config = config_manager.load_config()

        log_count = len(self.log_handler.buffer)

        self.assertEqual(type(config), dict)
        self.assertEqual(log_count, 8)

        self.assertEqual(self.log_handler.buffer[0].getMessage(), "Checking input parameters")
        self.assertEqual(self.log_handler.buffer[-1].getMessage(), "Config successfully loaded")