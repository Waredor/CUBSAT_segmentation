import  os
import logging

from ultralytics import YOLO

class ModelExporter:
    """
    Класс ModelExporter отвечает за сохранение обученной модели.
    Parameters:
        model (ultralytics.models.yolo.model.YOLO): обученная модель YOLOv11.
        output_dir (str): путь к директории для сохранения обученной модели.
        logger (logging.Logger): объект логгера.
    """
    def __init__(self, model: YOLO, output_dir: str, logger: logging.Logger):
        self.model = model
        self.output_dir = output_dir
        self.logger = logger

    def save_model(self, model_filename: str) -> None:
        """
        Метод save_model() осуществляет сохранение обученной модели с указанным именем
        файла.
        Parameters:
             model_filename (str): имя файла сохраняемой модели.
        Returns:
            None
        """
        self.logger.info("Saving model...")
        self.model.save(os.path.join(self.output_dir, model_filename))
        self.logger.info("Model saved")