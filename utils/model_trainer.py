from torch import cuda
from ultralytics import YOLO


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

    def __init__(self, model_cfg: str, hyperparameters: dict, logger) -> None:
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

    def train_model(self) -> YOLO:
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
        if device == 0:
            if cuda.is_available() is True:
                self.logger.info("Cuda is available, training on GPU")

            else:
                device = "cpu"
                self.logger.info("Cuda is not available, training on CPU")

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