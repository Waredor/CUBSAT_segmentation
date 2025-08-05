import os
import logging
import yaml

from types import UnionType
from dataclasses import dataclass
from pathlib import Path
from logging.handlers import RotatingFileHandler


@dataclass
class ParamConfig:
    """
    Класс ParamConfig отвечает за создание списка с метаданными о параметрах,
    поступающих в класс ConfigManager для валидации
    Parameters:
        name (str): имя параметра.
        expected_type (type): ожидаемый тип данных параметра.
        is_file (bool): флаг, сигнализирующий о том, является ли параметр путем к файлу.
        is_dir (bool): флаг, сигнализирующий о том, является ли параметр путем к директории.
        extensions (list): список допустимых расширений файла, которым является параметр.
            (если параметр это не файл - единственный элемент списка пустая строка)
    """
    name: str
    expected_type: type | UnionType
    is_file: bool
    is_dir: bool
    extensions: list


class ConfigManager:
    """
    Класс ConfigManager отвечает за загрузку и валидацию конфигурационных файлов
    и гиперпараметров модели.
    При создании класса инициализируется список с метаданными для валидации со следующей структурой:
     - Элементы с индексами 0 и 1: переменные, подаваемые на вход класса.
     - Элементы с индексами 2 - 7: значения, получаемые по ключам первого уровня словаря конфигурации пайплайна.
     - Элементы с индексами 8 - 10: значения, получаемые по ключам словаря, являющегося значением ключа первого уровня 'paths'.
     - Элементы с индексами 11 - 12: значения, получаемые по ключам словаря, являющегося значением ключа первого уровня 'names'.
     - Элементы с индексами 13 - 14: значения, получаемые по ключам словаря, являющегося значением ключа первого уровня 'extensions'.
     - Элементы с индексами 15 - 31: значения, получаемые по ключам словаря, являющегося значением ключа первого уровня 'model_hyperparameters'.
     - Элементы с индексами 32 - 36: значения, получаемые по ключам словаря, являющегося значением ключа первого уровня 'dataset_cfg'.
    Parameters:
        pipeline_cfg (str): путь до .yaml файла конфигурации пайплайна.
        project_root (str): путь до корневой папки проекта.
        logger (logging.Logger): объект логгера.
    """

    def __init__(self, project_root: str, pipeline_cfg: str,
                 logger: logging.Logger) -> None:
        self.params = {'project_root': project_root, 'pipeline_cfg': pipeline_cfg}
        self.metadata = [
            ParamConfig('project_root', str, False, True, ['']),    # Входные переменные
            ParamConfig('pipeline_cfg', str, True, False, ['.yaml']),
            ParamConfig('paths', dict, False, False, ['']), # Значения, получаемые по ключам первого уровня
            ParamConfig('names', dict, False, False, ['']),
            ParamConfig('extensions', dict, False, False, ['']),
            ParamConfig('model_hyperparameters', dict, False, False, ['']),
            ParamConfig('dataset_cfg', dict, False, False, ['']),
            ParamConfig('stages', list, False, False, ['']),
            ParamConfig('raw_data_dir', str, False, True, ['']),    # Значения, получаемые по ключам словаря 'paths'
            ParamConfig('inference_images_dir', str, False, True, ['']),
            ParamConfig('labelme_inference_annotations_dir', str, False, True, ['']),
            ParamConfig('pt_model_name', str, True, False, ['.pt']),    # Значения, получаемые по ключам словаря 'names'
            ParamConfig('exported_model_name', str, True, False, ['.pt']),
            ParamConfig('images_extensions', list, False, False, ['']), # Значения, получаемые по ключам словаря 'extensions'
            ParamConfig('labels_extensions', list, False, False, ['']),
            ParamConfig('epochs', int, False, False, ['']), # Значения, получаемые по ключам словаря 'model_hyperparameters'
            ParamConfig('imgsz', int, False, False, ['']),
            ParamConfig('batch', int, False, False, ['']),
            ParamConfig('lr0', float, False, False, ['']),
            ParamConfig('lrf', float, False, False, ['']),
            ParamConfig('patience', int, False, False, ['']),
            ParamConfig('device', int | str, False, False, ['']),
            ParamConfig('optimizer', str, False, False, ['']),
            ParamConfig('freeze_layers', int, False, False, ['']),
            ParamConfig('num_workers', int, False, False, ['']),
            ParamConfig('dropout', float, False, False, ['']),
            ParamConfig('weight_decay', float, False, False, ['']),
            ParamConfig('label_smoothing', float, False, False, ['']),
            ParamConfig('warmup_epochs', int, False, False, ['']),
            ParamConfig('iou', float, False, False, ['']),
            ParamConfig('cos_lr', bool, False, False, ['']),
            ParamConfig('augment_params', dict, False, False, ['']),
            ParamConfig('path', str, False, True, ['']), # Значения, получаемые по ключам словаря 'dataset_cfg'
            ParamConfig('train', str, False, False, ['']),
            ParamConfig('val', str, False, False, ['']),
            ParamConfig('nc', int, False, False, ['']),
            ParamConfig('names', list, False, False, ['']),
        ]
        self.logger = logger

    def _validate_param(self, param , config: ParamConfig) -> None:
        """
        Вспомогательный метод _validate_param() проверяет тип параметра, корректность путей
        к файлу/директории и расширение файла, если параметр является файлом либо директорией.
        Parameters:
            param: проверяемый параметр.
            config (ParamConfig): датакласс с метаданными для проверки валидности параметра.
        Returns:
            None
        Raises:
            ValueError: если параметр является отрицательным числом, пустой строкой либо None.
            TypeError: если тип параметра не соответствует требуемому.
            FileNotFoundError: если параметр не является путем к файлу.
            NotADirectoryError: если параметр не является путем к директории.
        """
        if param is None:
            self.logger.error(f"Parameter {config.name} is None")
            raise ValueError(f"Parameter {config.name} is None")

        if not isinstance(param, config.expected_type):
            self.logger.error(
                f"Wrong type of the {config.name} (expected: {config.expected_type}, got: {type(param)})")
            raise TypeError(f"Wrong type of the {config.name}")

        if isinstance(param, int | float):
            if param < 0:
                self.logger.error(f"Value for key {config.name} in YAML file is negative")
                raise ValueError(f"Value for key {config.name} in YAML file is negative")

        if isinstance(param, str):
            if len(param) == 0:
                self.logger.error(f"List element {param} for key {config.name} "
                                  f"is an empty string")
                raise ValueError(f"List element {param} for key {config.name} "
                                 f"is an empty string")

        if config.is_file:
            if not os.path.isfile(param):
                self.logger.error(f'{param} is not a path to file')
                raise FileNotFoundError(f'{param} is not a path to file')

            if not Path(param).suffix in config.extensions:
                self.logger.error(f'{param} has invalid file extension')
                raise ValueError(f'{param} has invalid file extension')

        elif config.is_dir:
            if not os.path.isdir(param):
                self.logger.error(f'{param} is not a directory')
                raise NotADirectoryError(f'{param} is not a directory')

    def check_input(self) -> None:
        """
        Метод check_input() проверяет входные переменные на соответствие требуемой конфигурации.
        Returns:
            None
        Raises:
            ValueError: если в словаре с входными переменными допущена ошибка.
        """
        self.logger.info("Starting checking input")
        input_metadata = self.metadata[0:2]
        for el in input_metadata:
            if el.name not in self.params.keys():
                raise ValueError(f"Config error! {el.name} is missing!")
            self._validate_param(self.params[el.name], el)
        self.logger.info("Checking input completed")

    def load_yaml(self) -> dict:
        """
        Метод load_yaml() загружает содержимое .yaml файла в словарь python
        Returns:
            data (dict): словарь с содержимым .yaml файла.
        Raises:
            ValueError: если синтаксис .yaml файла содержит ошибки
        """
        try:
            with open(self.params['pipeline_cfg'], mode='r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data

        except yaml.YAMLError as exc:
            self.logger.error(f"Error parsing YAML file {self.params['pipeline_cfg']}")
            raise ValueError(f"Error parsing YAML file {self.params['pipeline_cfg']}") from exc

    def _check_yaml_file(self, yaml_data: dict) -> None:
        """
        Вспомогательный метод _check_yaml_file() осуществляет проверку .yaml файла
        конфигурации пайплайна на корректность.
        Parameters:
            yaml_data (dict): словарь с данными .yaml файла конфигурации пайплайна
        Returns:
            None
        Raises:
            KeyError: если в файле нет требуемого ключа.
            ValueError: если значение, получаемое по ключу отрицательное,
                является пустой строкой или None.
            TypeError: если тип значения, получаемого по ключу,
                не соответствует требуемому.
            NotADirectoryError: если путь не является путем к дирректории.
            FileNotFoundError: если путь к директории не существует
        """
        self.logger.info("Starting validation of YAML file")
        for idx, param in enumerate(self.metadata):
            if 2 <= idx < 8:
                if param.name not in yaml_data.keys():
                    self.logger.error(f"Key {param.name} not found in pipeline_config YAML file")
                    raise KeyError(f"Key {param.name} not found in pipeline_config YAML file")
                self._validate_param(yaml_data[param.name], param)

            elif 8 <= idx < 11:
                if param.name not in yaml_data['paths'].keys():
                    self.logger.error(f"Key {param.name} not found in pipeline_config['paths'] dict")
                    raise KeyError(f"Key {param.name} not found in pipeline_config['paths'] dict")
                self._validate_param(yaml_data['paths'][param.name], param)

            elif idx == 11:
                if param.name not in yaml_data['names'].keys():
                    self.logger.error(f"Key {param.name} not found in pipeline_config['names'] dict")
                    raise KeyError(f"Key {param.name} not found in pipeline_config['names'] dict")
                full_path = os.path.join(
                    self.params['project_root'], yaml_data['names'][param.name]
                )
                self._validate_param(full_path, param)

            elif 13 <= idx < 15:
                if param.name not in yaml_data['extensions'].keys():
                    self.logger.error(f"Key {param.name} not found in pipeline_config['extensions'] dict")
                    raise KeyError(f"Key {param.name} not found in pipeline_config['extensions'] dict")
                self._validate_param(yaml_data['extensions'][param.name], param)

            elif 15 <= idx < 32:
                if param.name not in yaml_data['model_hyperparameters'].keys():
                    self.logger.error(f"Key {param.name} not found in pipeline_config['model_hyperparameters'] dict")
                    raise KeyError(f"Key {param.name} not found in pipeline_config['model_hyperparameters'] dict")
                self._validate_param(yaml_data['model_hyperparameters'][param.name], param)

            elif 32 <= idx < 37:
                if param.name not in yaml_data['dataset_cfg'].keys():
                    self.logger.error(f"Key {param.name} not found in pipeline_config['dataset_cfg'] dict")
                    raise KeyError(f"Key {param.name} not found in pipeline_config['dataset_cfg'] dict")

                if param.name not in ('train', 'val'):
                    if not isinstance(yaml_data['dataset_cfg'][param.name], list):
                        self._validate_param(yaml_data['dataset_cfg'][param.name], param)
                    else:
                        if len(set(yaml_data['dataset_cfg'][param.name])) != yaml_data['dataset_cfg']["nc"]:
                            self.logger.error("Length of class names list for key "
                                              "'names' does not match "
                                              "the number of classes specified in YAML, "
                                              "or class names are duplicated")
                            raise ValueError("Length of class names list for key "
                                             "'names' does not match "
                                             "the number of classes specified in YAML, "
                                             "or class names are duplicated")

                        for el in yaml_data['dataset_cfg'][param.name]:
                            if not isinstance(el, str):
                                self.logger.error(f"Type of list element {el} "
                                                  f"for key {param.name} does not match required type "
                                                  f"(expected: {str}, got: {type(el)})")
                                raise TypeError(f"Type of list element {el} "
                                                f"for key {param.name} does not match required type "
                                                f"(expected: {str}, got: {type(el)})")

                            if len(el) == 0:
                                self.logger.error(f"List element {el} for key {param.name} "
                                                  f"is an empty string")
                                raise ValueError(f"List element {el} for key {param.name} "
                                                 f"is an empty string")
                else:
                    full_path = os.path.join(
                        yaml_data['dataset_cfg']['path'], yaml_data['dataset_cfg'][param.name]
                    )
                    if not os.path.exists(full_path):
                        self.logger.error(f"{full_path} is not exists")
                        raise FileNotFoundError(f"{full_path} is not exists")
                    if not os.path.isdir(full_path):
                        self.logger.error(f"{full_path} is not a directory")
                        raise NotADirectoryError(f"{full_path} is not a directory")

        self.logger.info("Validated successfully")

    def load_config(self) -> dict:
        """
        Метод load_config() осуществляет загрузку конфигурационых файлов
        и гиперпараметров модели.
        Returns:
            hyperparameters (dict): словарь с путями конфигурационных файлов
                и гиперпараметрами модели.
        Raises:
            ValueError: если возникает ошибка в парсинге .yaml и json файлов.
        """
        hyperparameters = {}
        self.logger.info("Checking input parameters")
        self.check_input()

        self.logger.info("Loading YAML data")
        yaml_data = self.load_yaml()

        self.logger.info("Checking YAML file")
        self._check_yaml_file(yaml_data)

        for key, value in yaml_data.items():
            hyperparameters[key] = value

        self.logger.info("Config successfully loaded")
        return hyperparameters


def setup_logger(
        logger_name: str,
        logger_file_path: str,
        use_file_handler: bool = True
) -> logging.Logger:
    """
    Метод setup_logger() создает объект logging.Logger для логирования работы пайплайна
    Parameters:
        logger_name (str): имя создаваемого логгера.
        logger_file_path (str): путь к .txt файлу лога.
        use_file_handler (bool): флаг, отвечающий за создание файла лога.
    Returns:
        logger (logging.Logger): объект логгера.
    """
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s'
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if use_file_handler:
            os.makedirs(os.path.dirname(logger_file_path), exist_ok=True)
            file_handler = RotatingFileHandler(
                logger_file_path,
                maxBytes=1048576,
                backupCount=3
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger