import os
import logging
import json

from pathlib import Path
from logging.handlers import RotatingFileHandler

import yaml


class ConfigManager:
    """
    Класс ConfigManager отвечает за загрузку и валидацию конфигурационных файлов
    и гиперпараметров модели.
    Parameters:
        data_cfg (str): путь до .yaml файла конфигурации датасета
            в формате совместимом с моделью YOLOv11.
            Файл находится в корневой папке датасета.
            Также в корневой папке датасета находятся папки images/ и labels/.
        model_hyperparameters (str): путь до .json файла с гиперпараметрами
            для обучения модели YOLOv11.
        data_dir (str): путь до корневой папки датасета.
        model_cfg (str): путь до .yaml файла конфигурации модели,
            либо до .pt файла с предобученной моделью.
        output_dir (str): путь к директории для сохранения обученной модели.
        logger (logging.Logger): объект логгера.
    """

    def __init__(self, data_cfg: str, model_hyperparameters: str,
                 data_dir: str, model_cfg: str, output_dir: str,
                 logger: logging.Logger) -> None:
        self.params = [data_cfg, model_hyperparameters, data_dir, model_cfg, output_dir]
        self.metadata = {0: {'name': 'data_cfg', 'expected_type': str,
                             'is_file': True, 'is_dir': False, 'extension': ['.yaml']},
                         1: {'name': 'model_hyperparameters', 'expected_type': str,
                             'is_file': True, 'is_dir': False, 'extension': ['.json']},
                         2: {'name': 'data_dir', 'expected_type': str,
                             'is_file': False, 'is_dir': True, 'extension': ['']},
                         3: {'name': 'model_cfg', 'expected_type': str,
                             'is_file': True, 'is_dir': False, 'extension': ['.pt', '.yaml']},
                         4: {'name': 'output_dir', 'expected_type': str,
                             'is_file': False, 'is_dir': True, 'extension': ['']}
                         }
        self.logger = logger

    def _validate_path(self, el: str, is_file: bool, is_dir: bool, extensions: list) -> None:
        """
        Вспомогательный метод _validate_path() проверяет корректность путей
        к файлу/директории и расширение файла.
        Parameters:
            el (str): проверяемый путь
            is_file (bool): флаг, отвечающий за то,
                является ли данный путь путем к файлу
            is_dir (bool): флаг, отвечающий за то,
                является ли данный путь путем к директории
            extensions (list): список с расширениями файла
                (если это путь к директории, то расширение - список с пустой строкой
                в качестве единственного элемента)
        Returns:
            None
        Raises:
            NotADirectoryError: если путь, указанный как путь к директории,
                не является таковым
            FileNotFoundError: если путь, указанный как путь к файлу,
                не является таковым
            ValueError: если файл имеет неверное расширение
        """
        if is_dir:
            if not os.path.isdir(el):
                self.logger.error(f'{el} is not a directory')
                raise NotADirectoryError(f'{el} is not a directory')

        elif is_file:
            if not os.path.isfile(el):
                self.logger.error(f'{el} is not a path to file')
                raise FileNotFoundError(f'{el} is not a path to file')

            if not Path(el).suffix in extensions:
                self.logger.error(f'{el} has invalid file extension')
                raise ValueError(f'{el} has invalid file extension')

    def _check_json_file(self, json_dict) -> None:
        """
        Вспомогательный метод _check_json_file() осуществляет проверку .json файла
        с гиперпараметрами модели для обучения.
        Если обнаружены несоответствия типов данных значений словаря
        с ожидаемыми типами данных, либо какие-то из пар ключ-значение
        отсутствуют - метод вызывает ошибку.
        Parameters:
            json_dict (dict): словарь с именами и значениями гиперпараметров.
        Returns:
            None
        Raises:
            KeyError: если в файле нет требуемого ключа.
            ValueError: если значение, получаемое по ключу отрицательное или None.
            TypeError: если тип значения, получаемого по ключу,
                не соответствует требуемому.
        """
        json_metadata = {"epochs": [int],
                         "imgsz": [int],
                         "batch": [int],
                         "lr0": [float],
                         "patience": [int],
                         "device": [str, int],
                         "optimizer": [str],
                         "freeze_layers": [int]
                         }

        for key, value in json_metadata.items():
            if key not in json_dict.keys():
                self.logger.error(f"Key {key} not found in hyperparameters JSON file")
                raise KeyError(f"Key {key} not found in hyperparameters JSON file")

            if json_dict[key] is None:
                self.logger.error(f"Value for key {key} in JSON is None")
                raise ValueError(f"Value for key {key} in JSON is None")

            if type(json_dict[key]) not in value:
                self.logger.error(f"Type of value for key {key} "
                                  f"in JSON does not match required type "
                                  f"(expected: {value}, got: {type(json_dict[key])})")
                raise TypeError(f"Type of value for key {key} "
                                f"in JSON does not match required type "
                                f"(expected: {value}, got: {type(json_dict[key])})")

            if value in ([int], [float]):
                if json_dict[key] < 0:
                    self.logger.error(f"Value for key {key} in JSON file is negative")
                    raise ValueError(f"Value for key {key} in JSON file is negative")

            if key == "device":
                if json_dict[key] not in [0, "cpu"]:
                    self.logger.error(f"Invalid value for key {key} in JSON "
                                      f"(expected: {[0, 'cpu']}, got: {json_dict[key]})")
                    raise ValueError(f"Invalid value for key {key} in JSON "
                                     f"(expected: {[0, 'cpu']}, got: {json_dict[key]})")

    def _check_yaml_file(self, yaml_dict) -> None:
        """
        Вспомогательный метод _check_yaml_file() осуществляет проверку .yaml файла
        конфигурации датасета на корректность.
        Parameters:
            yaml_dict (dict): словарь, полученный при открытии .yaml файла
                через .yaml.safe_load().
        Returns:
            None
        Raises:
            KeyError: если в файле нет требуемого ключа.
            ValueError: если значение, получаемое по ключу отрицательное,
                является пустой строкой или None.
            TypeError: если тип значения, получаемого по ключу,
                не соответствует требуемому.
            NotADirectoryError: если путь к директории не существует или этот путь
                не является путем к дирректории.
        """
        yaml_metadata = {"path": str,
                         "train": str,
                         "val": str,
                         "nc": int,
                         "names": list
                         }

        for key, value in yaml_metadata.items():
            if key not in yaml_dict.keys():
                self.logger.error(f"Key {key} not found in dataset YAML file")
                raise KeyError(f"Key {key} not found in dataset YAML file")

            if yaml_dict[key] is None:
                self.logger.error(f"Value for key {key} in YAML is None")
                raise ValueError(f"Value for key {key} in YAML is None")

            if not isinstance(yaml_dict[key], value):
                self.logger.error(f"Type of value for key {key} in YAML "
                                  f"does not match required type "
                                  f"(expected: {value}, got: {type(yaml_dict[key])})")
                raise TypeError(f"Type of value for key {key} in YAML "
                                f"does not match required type "
                                f"(expected: {value}, got: {type(yaml_dict[key])})")

            if isinstance(yaml_dict[key], int):
                if yaml_dict[key] < 0:
                    self.logger.error(f"Value for key {key} in YAML file is negative")
                    raise ValueError(f"Value for key {key} in YAML file is negative")

            if isinstance(yaml_dict[key], list):
                if len(set(yaml_dict[key])) != yaml_dict["nc"]:
                    self.logger.error("Length of class names list for key "
                                      "'names' does not match "
                                      "the number of classes specified in YAML, "
                                      "or class names are duplicated")
                    raise ValueError("Length of class names list for key "
                                     "'names' does not match "
                                     "the number of classes specified in YAML, "
                                     "or class names are duplicated")

                for el in yaml_dict[key]:
                    if not isinstance(el, str):
                        self.logger.error(f"Type of list element {el} "
                                          f"for key {key} does not match required type "
                                          f"(expected: {str}, got: {type(el)})")
                        raise TypeError(f"Type of list element {el} "
                                        f"for key {key} does not match required type "
                                        f"(expected: {str}, got: {type(el)})")

                    if len(el) == 0:
                        self.logger.error(f"List element {el} for key {key} "
                                          f"is an empty string")
                        raise ValueError(f"List element {el} for key {key} "
                                         f"is an empty string")

            elif key in ('train', 'val'):
                full_path = os.path.join(self.params[2], yaml_dict[key])
                if not os.path.exists(full_path):
                    self.logger.error(f"{full_path} is not exists")
                    raise FileNotFoundError(f"{full_path} is not exists")

                if not os.path.isdir(full_path):
                    self.logger.error(f"{full_path} is not a directory")
                    raise NotADirectoryError(f"{full_path} is not a directory")

    def validate_config(self) -> None:
        """
        Метод validate_config() осуществляет проверку переданных в класс переменных
        на наличие различных ошибок.
        Returns:
            None
        Raises:
             ValueError: если тип переменной не соответствует целевому, либо None
        """
        self.logger.info("Starting validation")
        for idx, el in enumerate(self.params):
            if el is None:
                self.logger.error(f"Parameter {self.metadata[idx]['name']} is None")
                raise ValueError(f"Parameter {self.metadata[idx]['name']} is None")

            if not isinstance(el, self.metadata[idx]['expected_type']):
                self.logger.error(f"Parameter {self.metadata[idx]['name']} has incorrect type "
                                  f"(expected: {self.metadata[idx]['expected_type']}, "
                                  f"got: {type(el)})")
                raise ValueError(f"Parameter {self.metadata[idx]['name']} has incorrect type "
                                 f"(expected: {self.metadata[idx]['expected_type']}, "
                                 f"got: {type(el)})")

            if isinstance(el, str):
                self._validate_path(
                    el,
                    self.metadata[idx]['is_file'],
                    self.metadata[idx]['is_dir'],
                    self.metadata[idx]['extension']
                )

        self.logger.info("Validation completed")

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
        self.validate_config()
        hyperparameters = {}
        for idx, el in enumerate(self.params):
            if self.metadata[idx]['is_file'] is True:
                if Path(el).suffix == '.json':
                    try:
                        with open(el, mode='r', encoding='utf-8') as f:
                            train_hyperparameters = json.load(f)
                            self._check_json_file(train_hyperparameters)
                            for key in train_hyperparameters.keys():
                                hyperparameters[key] = train_hyperparameters[key]

                    except json.JSONDecodeError as exc:
                        self.logger.error(f"Error parsing JSON file {el}")
                        raise ValueError(f"Error parsing JSON file {el}") from exc

                # указываем индекс, так как расширение .yaml может иметь и файл модели
                elif Path(el).suffix == '.yaml' and idx == 0:
                    try:
                        with open(el, mode='r', encoding='utf-8') as f:
                            data_dict = yaml.safe_load(f)
                            self._check_yaml_file(data_dict)
                            hyperparameters['data_path'] = el
                            hyperparameters['class_names'] = data_dict['names']

                    except yaml.YAMLError as exc:
                        self.logger.error(f"Error parsing YAML file {el}")
                        raise ValueError(f"Error parsing YAML file {el}") from exc

                self.logger.info(f"Loaded configuration file: {el}")

            if self.metadata[idx]['is_dir'] is True:
                if idx == 4:
                    hyperparameters['output_dir'] = self.params[4]

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