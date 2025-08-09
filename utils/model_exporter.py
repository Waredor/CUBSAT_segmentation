import logging

from ultralytics import YOLO

class ModelExporter:
    """
    Класс ModelExporter отвечает за сохранение обученной модели.
    Parameters:
        model (ultralytics.models.yolo.model.YOLO): обученная модель YOLOv11.
        logger (logging.Logger): объект логгера.
    """
    def __init__(self, model: YOLO, logger: logging.Logger):
        self.model = model
        self.logger = logger

    def save_model(self, model_filepath: str) -> None:
        """
        Метод save_model() осуществляет сохранение обученной модели с указанным именем
        файла.
        Parameters:
             model_filepath (str): полный путь к файлу сохраняемой модели.
        Returns:
            None
        """
        self.logger.info("Saving model...")
        self.model.save(model_filepath)
        self.logger.info("Model saved")