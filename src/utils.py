import os
import logging
import glob
import base64
import json
import yaml
import torch
import cv2
import numpy as np
import ultralytics

from logging.handlers import RotatingFileHandler
from pathlib import Path
from PIL import Image


LOG_FILE = "\\cubsat_log.txt"
LOGGER_NAME = "utils_logger"

def setup_logger(use_file_handler: bool = True) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if use_file_handler:
            os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
            file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1048576, backupCount=3)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


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
    """

    def __init__(self, data_cfg: str, model_hyperparameters: str,
                 data_dir: str, model_cfg: str, output_dir: str) -> None:
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
        self.logger = setup_logger()

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
                self.logger.error(f"Type of value for key {key} in JSON does not match required type "
                                  f"(expected: {value}, got: {type(json_dict[key])})")
                raise TypeError(f"Type of value for key {key} in JSON does not match required type "
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
                self.logger.error(f"Type of value for key {key} in YAML does not match required type "
                                  f"(expected: {value}, got: {type(yaml_dict[key])})")
                raise TypeError(f"Type of value for key {key} in YAML does not match required type "
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
                                  f"(expected: {self.metadata[idx]['expected_type']}, got: {type(el)})")
                raise ValueError(f"Parameter {self.metadata[idx]['name']} has incorrect type "
                                 f"(expected: {self.metadata[idx]['expected_type']}, got: {type(el)})")

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


class ModelTrainer:
    """
    Класс ModelTrainer отвечает за инициализацию и обучение модели YOLOv11.
    Parameters:
        model_cfg (str): путь к .yaml файлу конфигурации модели
            или .pt файлу предобученной модели.
        hyperparameters (dict): словарь с гиперпараметрами модели для обучения
            и путем к конфигурационному файлу датасета.
    """

    def __init__(self, model_cfg: str, hyperparameters: dict) -> None:
        self.model_cfg = model_cfg
        self.hyperparameters = hyperparameters
        self.model = ultralytics.YOLO(self.model_cfg)
        self.logger = setup_logger()

    def freeze_layers(self, num_layers_to_freeze: int) -> None:
        """
        Метод freeze_layers() осуществляет заморозку слоев в backbone.
        Parameters:
            num_layers_to_freeze (int): количество замораживаемых слоев начиная с входного.
        """
        layer_count = 0
        for param in self.model.model.parameters():
            if layer_count < num_layers_to_freeze:
                param.requires_grad = False
            else:
                break
            layer_count += 1
        self.logger.info(f"Froze first {layer_count} layers")

    def train_model(self) -> ultralytics.models.yolo.model.YOLO:
        """
        Метод train_model выполняет обучение модели YOLOv11.
        Returns:
            self.model (ultralytics.models.yolo.model.YOLO)
        """
        self.logger.info("Starting training")
        self.logger.info(f"Training model with parameters: {self.hyperparameters}")
        num_layers_to_freeze = self.hyperparameters['freeze_layers']
        self.freeze_layers(num_layers_to_freeze)
        data_dir = self.hyperparameters['data_path']
        epochs = self.hyperparameters['epochs']
        batch_size = self.hyperparameters['batch']
        image_size = self.hyperparameters['imgsz']
        initial_learning_rate = self.hyperparameters['lr0']
        optimizer = self.hyperparameters['optimizer']
        patience = self.hyperparameters['patience']
        device = self.hyperparameters['device']
        self.model.train(
            data=data_dir,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            lr0=initial_learning_rate,
            optimizer=optimizer,
            patience=patience,
            device=device
        )
        self.logger.info("Training completed")
        return self.model


class AnnotationProcessor:
    """
    Класс AnnotationProcessor создает аннотации в формате LabelMe .json
    к тестовым изображениям.
    Parameters:
        output_dir (str): директория для сохранения созданных аннотаций.
        class_names (list): список с именами используемых классов.
    """

    def __init__(self, output_dir: str, class_names: list) -> None:
        self.output_dir = output_dir
        self.class_names = class_names
        self.logger = setup_logger()

    def mask_to_polygons(self, mask: np.ndarray) -> list:
        """
        Метод mask_to_polygons преобразует маски объектов, полученные в результате инференса,
        в список полигонов.
        Parameters:
            mask (np.ndarray): numpy маски объектов, полученные в результате инференса.
        Returns:
            polygons (list): список полигонов объектов.
        """
        polygons = []
        if mask.ndim == 3:
            for i in range(mask.shape[0]):
                single_mask = mask[i].astype(np.uint8)
                contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        polygon = contour.squeeze().tolist()
                        polygons.append(polygon)
        else:
            single_mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) >= 3:
                    polygon = contour.squeeze().tolist()
                    polygons.append(polygon)

        return polygons

    def create_labelme_json(self, image_path: str, masks: np.ndarray, labels: np.ndarray,
                            class_names: list, output_dir: str) -> str:
        """
        Метод create_labelme_json() осуществляет создание аннотаций к инференсу
        из InferenceRunner в формате .json
        Parameters:
            image_path (str): путь к изображению, для которого создается аннотация.
            masks (np.ndarray): numpy массив масок полигонов объектов на изображении.
            labels (np.ndarray): numpy массив меток классов полигонов на изображении.
            class_names (list): список имен класов объектов, которые есть
                на изображениях в датасете.
            output_dir (str): выходная директория для сохранения .json аннотаций
                к изображениям.
        """
        if not os.path.exists(output_dir):
            self.logger.error(f"Incorrect directory {output_dir}")
            raise NotADirectoryError(f"Incorrect directory {output_dir}")

        if not os.path.isfile(image_path):
            self.logger.error(f"{image_path} is not a path to a file")
            raise FileNotFoundError(f"{image_path} is not a path to a file")

        if np.size(masks) == 0:
            self.logger.error(f"{masks} is an empty array")
            raise ValueError(f"{masks} is an empty array")

        if np.size(labels) == 0:
            self.logger.error(f"{labels} is an empty array")
            raise ValueError(f"{labels} is an empty array")

        image_name = Path(image_path).name
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        _, buffer = cv2.imencode(".jpg", image)
        image_data = base64.b64encode(buffer).decode("utf-8")

        labelme_data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_name,
            "imageData": image_data,
            "imageHeight": height,
            "imageWidth": width
        }

        for mask, label in zip(masks, labels):
            polygons = self.mask_to_polygons(mask.astype(np.uint8))
            for polygon in polygons:
                shape = {
                    "label": class_names[int(label)],
                    "points": polygon,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                labelme_data["shapes"].append(shape)

        output_path = os.path.join(
            output_dir,
            image_name.replace(".jpg", ".json").replace(".png", ".json")
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, mode="w", encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2)
        self.logger.info(f"Created JSON-file: {output_path}")
        return output_path


class InferenceRunner:
    """
    Класс InferenceRunner осуществляет инференс на тестовых изображениях.
    Parameters:
        model (torch.nn.Module): обученная модель YOLOv11.
        img_size (int): размер изображения (изображение квадратное).
        annotation_processor (AnnotationProcessor): экземпляр класса AnnotationProcessor
            для создания разметки к инференсу.
    """

    def __init__(self, model: torch.nn.Module, img_size: int,
                 annotation_processor: AnnotationProcessor) -> None:
        self.model = model
        self.img_size = img_size
        self.annotation_processor = annotation_processor
        self.logger = setup_logger()

    def run_inference(self, image_path: str) -> list:
        """
        Метод run_inference производит инференс для одного изображения,
        хранящегося по указанному пути.
        Parameters:
            image_path (str): путь к изображению для инференса.
        Returns:
            results (ultralytics.YOLO.results): объект класса ultralytics.YOLO.results
                с результатами инференса.
        """
        try:
            results = self.model.predict(image_path, imgsz=self.img_size, conf=0.5, iou=0.7)
            return results

        except RuntimeError as exc:
            self.logger.error("Internal model error! Runtime error")
            raise RuntimeError("Internal model error! Runtime error") from exc

        except FileNotFoundError as exc:
            self.logger.error(f"File {image_path} doesn't found")
            raise FileNotFoundError(f"File {image_path} doesn't found") from exc

    def process_images(self, test_images_dir: str) -> None:
        """
        Метод process_images() обрабатывает все изображения в указанной директории,
        выполняя инференс для каждого изображения и передавая результаты
        в AnnotationProcessor.
        Parameters:
            test_images_dir (str): Путь к директории с изображениями
        """
        if not os.path.isdir(test_images_dir):
            self.logger.error(f"{test_images_dir} is not a directory")
            raise NotADirectoryError(f"{test_images_dir} is not a directory")

        test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if
                       f.endswith(('.jpg', '.png'))]
        for image_path in test_images:
            results = self.run_inference(image_path)
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()
                self.annotation_processor.create_labelme_json(
                    image_path=image_path,
                    masks=masks,
                    labels=labels,
                    class_names=self.annotation_processor.class_names,
                    output_dir=self.annotation_processor.output_dir
                )
            else:
                self.logger.warning(f"No objects in {image_path}")


class Pipeline:
    """
    Класс Pipeline предназначен для автоматизации процессов сбора,
    обработки и аугментации данных,
    обучения, валидации и сохранения моделей машинного обучения.
    Parameters:
        data_cfg (str): .yaml файл конфигурации датасета в формате YOLOv11.
        data_dir (str): путь к директории с датасетом в файловой системе.
        model_cfg (str): путь к .pt файлу предобученной модели YOLOv11 в файловой системе.
        model_hyperparameters (str): путь к .json файлу с гиперпараметрами модели.
        output_dir (str): путь к директории сохранения обученной модели.
    """

    def __init__(self, data_cfg: str, model_cfg: str, model_hyperparameters: str, data_dir: str,
                 output_dir: str) -> None:
        self.logger = setup_logger()
        self.logger.info("Pipeline init")
        self.config_manager = ConfigManager(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            model_hyperparameters=model_hyperparameters,
            data_dir=data_dir,
            output_dir=output_dir
        )
        self.config = self.config_manager.load_config()
        self.data_dir = data_dir
        self.model_trainer = ModelTrainer(
            model_cfg=model_cfg,
            hyperparameters=self.config
        )
        self.model = self.model_trainer.model
        self.logger.info("Pipeline init successfully")

    def fine_tune_for_labeling(self, model_filename: str) -> None:
        """
        Метод fine_tune_for_labeling() используется для дообучения под конкретную задачу модели,
        предобученной на больших данных и ее сохранения в указанной директории.
        Parameters:
            model_filename (str): имя файла обученной модели с расширением .pt.
        """
        self.logger.info("Starting fine-tuning")
        self.model = self.model_trainer.train_model()
        self.logger.info("Saving fine-tuned model")
        self.model.save(self.config['output_dir'] + model_filename)
        self.logger.info("Successfully saved")

    def create_new_json_annotations(self, test_images_dir: str,
                                    annotations_output_dir: str) -> None:
        """
        Метод create_new_annotations() используется для создания разметки тестовых изображений
        с использованием предобученной модели.
        Parameters:
            test_images_dir (str): директория с изображениями для инференса.
            annotations_output_dir (str): директория для сохранения файлов разметки.
        """
        self.logger.info("Starting creating annotations")
        annotation_processor = AnnotationProcessor(
            class_names=self.config['class_names'],
            output_dir=annotations_output_dir
        )
        inference_runner = InferenceRunner(
            model=self.model,
            img_size=self.config['imgsz'],
            annotation_processor=annotation_processor
        )
        inference_runner.process_images(test_images_dir)
        self.logger.info("Creating annotations finished")

    def convert_labelme_to_yolo(self, labelme_annotations_path: str,
                                yolo_annotations_path: str) -> None:
        """
        Метод convert_labelme_to_yolo() конвертирует аннотации изображений в формате .json LabelMe
        в формат YOLOv11 .txt
        Parameters:
            labelme_annotations_path (str): путь к директории с аннотациями в формате .json LabelMe.
            yolo_annotations_path (str): путь к директории с аннотациями в формате .txt YOLOv11.
        Raises:
            NotADirectoryError: если путь не является директорией
            FileNotFoundError: если файл по искомому пути не найден
        """
        if not os.path.isdir(labelme_annotations_path):
            self.logger.error(f"{labelme_annotations_path} is not a directory")
            raise NotADirectoryError(f"{labelme_annotations_path} is not a directory")

        if not os.path.isdir(yolo_annotations_path):
            self.logger.error(f"{yolo_annotations_path} is not a directory")
            raise NotADirectoryError(f"{yolo_annotations_path} is not a directory")

        class_map = {name: idx for idx, name in enumerate(self.config['class_names'])}

        os.makedirs(yolo_annotations_path, exist_ok=True)
        json_files = glob.glob(os.path.join(labelme_annotations_path, "*.json"))
        if len(json_files) == 0:
            self.logger.warning(f"There are no annotations in {labelme_annotations_path}")
            raise FileNotFoundError(f"There are no annotations in {labelme_annotations_path}")

        self.logger.info("Starting annotations convertation")
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            except json.decoder.JSONDecodeError:
                self.logger.warning(f"Empty file {f}")
                continue

            image_filename = data.get('imagePath')
            image_path = os.path.join(labelme_annotations_path, image_filename)

            if not os.path.exists(image_path):
                self.logger.warning(f"There are no images in: {image_path}")
                continue

            with Image.open(image_path) as img:
                w, h = img.size

            yolo_lines = []
            for shape in data['shapes']:
                label = shape.get('label')
                points = shape.get('points', [])

                if label not in class_map:
                    self.logger.warning(f"Skipped class: '{label}' в {json_path}")
                    continue

                if len(points) < 3:
                    self.logger.warning(f"Skipped object with a few points {json_path}")
                    continue

                class_id = class_map[label]

                norm_points = []
                for x, y in points:
                    norm_x = round(x / w, 6)
                    norm_y = round(y / h, 6)
                    norm_points.extend([norm_x, norm_y])

                yolo_line = f"{class_id} " + " ".join(map(str, norm_points))
                yolo_lines.append(yolo_line)

            base_name = os.path.splitext(os.path.basename(json_path))[0]
            txt_path = os.path.join(yolo_annotations_path, base_name + ".txt")

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(yolo_lines))
                self.logger.info(f"Converted {base_name}.txt")

        self.logger.info("Annotations convertation is finished")