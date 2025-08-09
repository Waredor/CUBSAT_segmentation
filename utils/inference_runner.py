import logging
import os

from ultralytics import YOLO

class InferenceRunner:
    """
    Класс InferenceRunner осуществляет инференс на тестовых изображениях.
    Parameters:
        model (torch.nn.Module): обученная модель YOLOv11.
        img_size (int): размер изображения (изображение квадратное).
            для создания разметки к инференсу.
        logger (logging.Logger): объект логгера.
    """

    def __init__(
            self,
            model: YOLO,
            img_size: int,
            logger: logging.Logger
    ) -> None:
        self.model = model
        self.img_size = img_size
        self.logger = logger

    def run_inference(self, image_path: str, batch_size: int, confidence: float, iou: float) -> list:
        """
        Метод run_inference производит инференс для одного изображения,
        хранящегося по указанному пути.
        Parameters:
            image_path (str): путь к изображению для инференса.
            batch_size (int): Размер батча
            confidence (float): параметр confidence модели
            iou (float): пороговое значение IoU для предсказаний
        Returns:
            results (list): объект с результатами инференса.
        """
        try:
            results = self.model.predict(
                image_path, imgsz=self.img_size, conf=confidence, iou=iou, batch=batch_size
            )
            return results

        except RuntimeError as exc:
            self.logger.error("Internal model error! Runtime error")
            raise RuntimeError("Internal model error! Runtime error") from exc

        except FileNotFoundError as exc:
            self.logger.error(f"File {image_path} doesn't found")
            raise FileNotFoundError(f"File {image_path} doesn't found") from exc

    def process_images(self, test_images_dir: str, batch_size: int, confidence: float, iou: float) -> list:
        """
        Метод process_images() обрабатывает все изображения в указанной директории,
        выполняя инференс для каждого изображения.
        Parameters:
            test_images_dir (str): Путь к директории с изображениями
            batch_size (int): Размер батча
            confidence (float): параметр confidence модели
            iou (float): пороговое значение IoU для предсказаний
        Returns:
            inference_results (list): Список словарей с именами файлов изображений,
                масками и метками
        """
        if not os.path.isdir(test_images_dir):
            self.logger.error(f"{test_images_dir} is not a directory")
            raise NotADirectoryError(f"{test_images_dir} is not a directory")

        inference_results = []
        for f in os.listdir(test_images_dir):
            if f.endswith(('.jpg', '.png')):
                image_path = os.path.join(test_images_dir, f)
                results = self.run_inference(image_path, batch_size, confidence, iou)
                if results[0].masks is not None:
                    inference_results.append({
                        "filename": f,
                        "masks": results[0].masks.data.cpu().numpy(),
                        "labels": results[0].boxes.cls.cpu().numpy()
                    })

                else:
                    self.logger.warning(f"No objects in {image_path}")

        return inference_results