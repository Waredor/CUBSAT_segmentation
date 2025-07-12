import torch
import os
import base64
import json
import cv2
import numpy as np
import yaml
import ultralytics
from pathlib import Path


class ConfigManager:
    """
    Класс ConfigManager отвечает за загрузку и валидацию конфигурационных файлов и гиперпараметров модели.

    Parameters:
        data_cfg (str): путь до .yaml файла конфигурации датасета в формате совместимом с моделью YOLOv11.
            Файл находится в корневой папке датасета. Также в корневой папке датасета находятся папки images/ и labels/.
        model_hyperparameters (str): путь до .json файла с гиперпараметрами для обучения модели YOLOv11.
        data_dir (str): путь до корневой папки датасета.
        model_cfg (str): путь до .yaml файла конфигурации модели, либо до .pt файла с предобученной моделью.
        output_dir (str): путь до папки
    """
    def __init__(self, data_cfg: str, model_hyperparameters: str, data_dir: str,
                 model_cfg: str, output_dir: str) -> None:
        self.params = [data_cfg, model_hyperparameters, data_dir, model_cfg, output_dir]
        self.metadata = {0: {'expected_type': str, 'is_file': True, 'is_dir': False, 'extension': '.yaml'},
                         1: {'expected_type': str, 'is_file': True, 'is_dir': False, 'extension': '.json'},
                         2: {'expected_type': str, 'is_file': False, 'is_dir': True, 'extension': ''},
                         3: {'expected_type': str, 'is_file': True, 'is_dir': False, 'extension': '.pt'},
                         4: {'expected_type': str, 'is_file': False, 'is_dir': True, 'extension': ''}
                         }
        self.hyperparameters = {
            "class_names": None,
            "epochs": None,
            "imgsz": None,
            "batch": None,
            "lr0": None,
            "patience": None,
            "device": None,
            "optimizer": None,
            "freeze_layers": None,
            "data_path": None
        }

    def _validate_path(self, el: str, is_file: bool, is_dir: bool, extension: str) -> None:
        """
        Вспомогательный метод _validate_path() проверяет корректность путей к файлу/директории и расширение файла.

        Parameters:
            el (str): проверяемый путь
            is_file (bool): флаг, отвечающий за то, является ли данный путь путем к файлу
            is_dir (bool): флаг, отвечающий за то, является ли данный путь путем к директории
            extension (str): расширение файла (если это путь к директории, то расширение - пустая строка)

        Raises:
            NotADirectoryError: если путь, указанный как путь к директории, не является таковым
            FileNotFoundError: если путь, указанный как путь к файлу, не является таковым
            ValueError: если файл имеет неверное расширение
        """
        if is_dir:
            if not os.path.isdir(el):
                raise NotADirectoryError(f'{el} не является путем к директории')

        elif is_file:
            if not os.path.isfile(el):
                raise FileNotFoundError(f'{el} не является путем к файлу')

            else:
                if Path(el).suffix != extension:
                    raise ValueError(f'{el} имеет неверное расширение файла')


    def _check_json_file(self, json_dict) -> None:
        """
        Вспомогательный метод _check_json_file() осуществляет проверку .json файла
        с гиперпараметрами модели для обучения. Если обнаружены несоответствия типов данных значений словаря
        с ожидаемыми типами данных, либо какие-то из пар ключ-значение отсутствуют - метод вызывает ошибку.
        Parameters:
            json_dict (dict): словарь с именами и значениями гиперпараметров.

        Raises:
            KeyError: если в файле нет требуемого ключа.
            ValueError: если значение, получаемое по ключу отрицательное или None.
            TypeError: если тип значения, получаемого по ключу, не соответствует требуемому.
        """
        json_metadata = {"epochs": int,
                    "imgsz": int,
                    "batch": int,
                    "lr0": float,
                    "patience": int,
                    "device": str, #заменить на int, если используется cuda
                    "optimizer": str,
                    "freeze_layers": int
                    }

        for key in json_metadata.keys():
            if key not in json_dict.keys():
                raise KeyError(f"в .json файле гиперпараметров модели нет ключа {key}")

            elif json_dict[key] is None:
                raise ValueError(f"Значение, получаемое из .json по ключу {key} None")

            elif json_metadata[key] != type(json_dict[key]):
                raise TypeError(f"Тип значения, получаемого из .json по ключу {key}, не соответствует требуемому типу"
                                f"(expected: {json_metadata[key]}, got: {type(json_dict[key])})")

            elif json_metadata[key] == int or json_metadata[key] == float:
                if json_dict[key] < 0:
                    raise ValueError(f"Значение, получаемое из .json файла по ключу {key} отрицательное")


    def _check_yaml_file(self, yaml_dict) -> None:
        """
        Вспомогательный метод _check_yaml_file() осуществляет проверку .yaml файла конфигурации датасета на корректность.

        Parameters:
            yaml_dict (dict): словарь, полученный при открытии .yaml файла через .yaml.safe_load().

        Raises:
            KeyError: если в файле нет требуемого ключа.
            ValueError: если значение, получаемое по ключу отрицательное, является пустой строкой или None.
            TypeError: если тип значения, получаемого по ключу, не соответствует требуемому.
            NotADirectoryError: если путь к директории не существует или этот путь не является путем к дирректории.
        """
        yaml_metadata = {"path": str,
                    "train": str,
                    "val": str,
                    "nc": int,
                    "names": list
                    }

        for key in yaml_metadata.keys():
            if key not in yaml_dict.keys():
                raise KeyError(f"в .yaml файле конфигурации датасета нет ключа {key}")

            elif yaml_dict[key] is None:
                raise ValueError(f"Значение, получаемое из .yaml по ключу {key} None")

            elif type(yaml_dict[key]) != yaml_metadata[key]:
                raise TypeError(f"Тип значения, получаемого из .yaml по ключу {key}, не соответствует требуемому типу"
                                f"(expected: {yaml_metadata[key]}, got: {type(yaml_dict[key])})")

            elif type(yaml_dict[key]) == int:
                if yaml_dict[key] < 0:
                    raise ValueError(f"Значение, получаемое из .yaml файла по ключу {key} отрицательное")

            elif type(yaml_dict[key]) == list:
                if len(set(yaml_dict[key])) != yaml_dict["nc"]:
                    raise ValueError(f"Длина списка с именами классов, получаемого по ключу 'names'"
                                     f"не соответствует указанному в .yaml файле количеству классов,"
                                     f"либо имена классов дублируются")

                for el in yaml_dict[key]:
                    if type(el) != str:
                        raise TypeError(f"Тип значения элемента списка {el}, получаемого по ключу {key}, не соответствует требуемому типу"
                                f"(expected: {str}, got: {type(el)}")

                    elif len(el) == 0:
                        raise ValueError(f"Элемент {el} списка, получаемого по ключу {key} является пустой строкой")

            elif key == 'train' or key == 'val':
                full_path = os.path.join(self.params[2], yaml_dict[key])
                if not os.path.exists(full_path): # тут возможно надо будет "объединить" путь по ключу
                    # и полный путь до .yaml файла
                    raise NotADirectoryError(f'Пути {full_path} не существует')

                elif not os.path.isdir(full_path):
                    raise NotADirectoryError(f'{full_path} не является директорией')


    def validate_config(self) -> None:
        """
        Метод validate_config() осуществляет проверку переданных в класс переменных на наличие различных ошибок.

        Raises:
             ValueError: если тип переменной не соответствует целевому, либо None
        """
        for idx, el in enumerate(self.params):
            if el is None:
                raise ValueError(f'Переменная {el} имеет тип данных None.')

            elif type(el) != self.metadata[idx]['expected_type']:
                    raise ValueError(f"Переменная {el} имеет неправильный тип данных"
                                     f" (expected: {self.metadata[idx]['expected_type']}, got: {type(el)}).")

            else:
                if type(el) == str:
                    self._validate_path(el, self.metadata[idx]['is_file'],
                                        self.metadata[idx]['is_dir'], self.metadata[idx]['extension'])


    def load_config(self) -> dict:
        """
        Метод load_config() осуществляет загрузку конфигурационых файлов и гиперпараметров модели.
        Returns:
            hyperparameters (dict): словарь с путями конфигурационных файлов и гиперпараметрами модели.
        """
        self.validate_config()
        for idx, el in enumerate(self.params):
            if self.metadata[idx]['is_file'] is True:
                if Path(el).suffix == '.json':
                    with open(el, 'r') as f:
                        train_hyperparameters = json.load(f)
                        self._check_json_file(train_hyperparameters)
                        for key in train_hyperparameters.keys():
                            self.hyperparameters[key] = train_hyperparameters[key]

                elif Path(el).suffix == '.yaml':
                    with open(el) as f:
                        data_dict = yaml.safe_load(f) # тут можно обработать исключения YAMLError
                        self._check_yaml_file(data_dict)
                        self.hyperparameters['data_path'] = el
                        self.hyperparameters['class_names'] = data_dict['names']

        return self.hyperparameters


class ModelTrainer:
    """
    Класс ModelTrainer отвечает за инициализацию и обучение модели YOLOv11.

    Parameters:
        model_cfg (str): путь к .yaml файлу конфигурации модели или .pt файлу предобученной модели.
        hyperparameters (dict): словарь с гиперпараметрами модели для обучения и путем к конфигурационному файлу датасета.
    """
    def __init__(self, model_cfg: str, hyperparameters: dict) -> None:
        self.model_cfg = model_cfg
        self.hyperparameters = hyperparameters
        self.model = ultralytics.YOLO(self.model_cfg)


    def _freeze_layers(self, num_layers_to_freeze: int) -> None:
        """
        Вспомогательный метод _freeze_layers() осуществляет заморозку слоев в backbone.

        Parameters:
            num_layers_to_freeze (int): количество замораживаемых слоев начиная с входного.
        """


    def train_model(self) -> torch.nn.Module:
        """
        Метод train_model выполняет обучение модели YOLOv11.

        Returns:
            self.model (torch.nn.Module)
        """
        num_layers_to_freeze = self.hyperparameters['freeze_layers']
        self._freeze_layers(num_layers_to_freeze)
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

        return self.model


class AnnotationProcessor:
    """
    Класс AnnotationProcessor создает аннотации в формате LabelMe .json к тестовым изображениям.
    Parameters:
        output_dir (str): директория для сохранения созданных аннотаций.
        class_names (list): список с именами используемых классов.
    """
    def __init__(self, output_dir: str, class_names: list) -> None:
        self.output_dir = output_dir
        self.class_names = class_names


    def mask_to_polygons(self, mask) -> list:
        """
        Метод mask_to_polygons преобразует маски объектов, полученные в результате инференса, в список полигонов.
        Parameters:
            mask : маски объектов, полученные в результате инференса.
        Returns:
            polygons (list): список полигонов объектов.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            if len(contour) >= 3:
                polygon = contour.squeeze().tolist()
                polygons.append(polygon)

        return polygons


    def create_labelme_json(self, image_path, masks, labels, class_names, output_dir) -> str:
        """

        """
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

        output_path = os.path.join(output_dir, image_name.replace(".jpg", ".json").replace(".png", ".json"))
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labelme_data, f, indent=2)
        return output_path


class InferenceRunner:
    """

    """
    def __init__(self, model: torch.nn.Module, img_size: int, data_dir: str,
                 annotation_processor: AnnotationProcessor) -> None:
        self.model = model
        self.img_size = img_size
        self.data_dir = data_dir
        self.annotation_processor = annotation_processor


    def run_inference(self, image_path: str) -> list:
        """
        Метод run_inference производит инференс для одного изображения, хранящегося по указанному пути.
        Parameters:
            image_path (str): путь к изображению для инференса.
        Returns:
            results (ultralytics.YOLO.results): объект класса ultralytics.YOLO.results с результатами инференса.
        """
        results = self.model.predict(image_path, imgsz=self.img_size, conf=0.5, iou=0.7)
        return results


    def process_images(self) -> None:
        """
        Метод process_images() обрабатывает все изображения в указанной директории,
        выполняя инференс для каждого изображения и передавая результаты в AnnotationProcessor.
        """
        test_images_dir = os.path.join(self.data_dir, "images/test")
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
                print(f"Нет объектов в {image_path}")


class ModelSaver:
    """
    Класс ModelSaver осуществляет сохранение модели после обучения в формате .pt.
    Parameters:
        model (torch.nn.Module): обученная модель для сохранения.
    """
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def save_model(self, model_output_dir: str) -> None:
        """
        Метод save_model() осуществлет сохранение модели в выбранной директории.
        Parameters:
            model_output_dir (str): путь к директории для сохранения обученной модели
        """
        self.model.save(model_output_dir)

class Pipeline:
    """
    Класс Pipeline предназначен для автоматизации процессов сбора, обработки и аугментации данных,
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
        self.config_manager = ConfigManager(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            model_hyperparameters=model_hyperparameters,
            data_dir=data_dir,
            output_dir=output_dir
        )
        self.config = self.config_manager.load_config()
        self.model_trainer = ModelTrainer(
            model_cfg=model_cfg,
            hyperparameters=self.config_manager.hyperparameters
        )
        self.annotation_processor = AnnotationProcessor(
            class_names=self.config_manager.hyperparameters['class_names'],
            output_dir=output_dir
        )
        self.inference_runner = InferenceRunner(
            model=self.model_trainer.train_model(),
            img_size=self.config_manager.hyperparameters['imgsz'],
            data_dir=data_dir,
            annotation_processor=self.annotation_processor
        )
        self.model_saver = ModelSaver(
            model=self.model_trainer.train_model()
        )


    def fine_tune_for_labeling(self, model_output_dir) -> None:
        """
        Метод fine_tune_for_labeling() используется для дообучения под конкретную задачу модели,
        предобученной на больших данных. В результате работы метода происходит последовательный вызов
        экземпляров классов ConfigManager, ModelTrainer, InferenceRunner и AnnotationProcessor внутри InferenceRunner.
        ConfigManager осуществляет проверку путей к файлам конфигурации и директориям, а также переменных в файлах конфигурации
        на валидность.
        ModelTrainer осуществляет инициализацию и обучение модели.
        InferenceRunner осуществляет инференс на тестовом наборе данных и вызывает AnnotationProcessor для
        создания аннотаций к инференсу в формате LabelMe .json.
        """
        self.config_manager.load_config()
        self.model_trainer.train_model()
        self.model_saver.save_model(model_output_dir)
        self.inference_runner.process_images()