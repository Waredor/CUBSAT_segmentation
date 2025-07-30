import logging

from ultralytics import YOLO
from torch import cuda


class ModelTrainer:
    """
    Класс ModelTrainer отвечает за инициализацию и обучение модели YOLOv11.
    Parameters:
        model_cfg (str): путь к .yaml файлу конфигурации модели
            или .pt файлу предобученной модели.
        hyperparameters (dict): словарь с гиперпараметрами модели для обучения
            и путем к конфигурационному файлу датасета.
        logger (logging.Logger): объект логгера.
    """

    def __init__(self, model_cfg: str, hyperparameters: dict, logger: logging.Logger) -> None:
        self.model_cfg = model_cfg
        self.hyperparameters = hyperparameters
        self.model = YOLO(self.model_cfg)
        self.logger = logger

    def freeze_layers(self, num_layers_to_freeze: int) -> None:
        """
        Метод freeze_layers() осуществляет заморозку слоев в backbone.
        Parameters:
            num_layers_to_freeze (int): количество замораживаемых слоев начиная с входного.
        Returns:
            None
        """
        layer_count = 0
        for param in self.model.model.parameters():
            if layer_count < num_layers_to_freeze:
                param.requires_grad = False
            else:
                break
            layer_count += 1
        self.logger.info(f"Froze first {layer_count} layers")

    def train_model(self, augment=False) -> YOLO:
        """
        Метод train_model выполняет обучение модели YOLOv11.
        Parameters:
            augment (bool): флаг использования аугментаций.
            с помощью albumentations.Compose()
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

        if device == 0:
            if cuda.is_available():
                self.logger.info("Using GPU device")

            else:
                self.logger.info("Using CPU device")
                device = "cpu"

        augment_params = {
            'hsv_h': 0.015,  # Аугментация оттенка (аналог RandomBrightnessContrast)
            'hsv_s': 0.7,  # Аугментация насыщенности
            'hsv_v': 0.4,  # Аугментация яркости
            'fliplr': 0.5,  # Горизонтальный флип (аналог HorizontalFlip(p=0.5))
            'degrees': 30,  # Поворот до 30 градусов (аналог Rotate(limit=30, p=0.3))
            'scale': 0.2,  # Масштабирование (приближение к RandomCrop)
            'erasing': 0.2  # Случайное стирание (приближение к GaussNoise)
        } if augment else {}

        self.model.train(
            data=data_dir,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            lr0=initial_learning_rate,
            optimizer=optimizer,
            patience=patience,
            device=device,
            **augment_params
        )

        self.logger.info("Training completed")
        return self.model