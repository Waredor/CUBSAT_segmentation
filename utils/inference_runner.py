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

    def run_inference(self, image_path: str) -> list:
        """
        Метод run_inference производит инференс для одного изображения,
        хранящегося по указанному пути.
        Parameters:
            image_path (str): путь к изображению для инференса.
        Returns:
            results (list): объект с результатами инференса.
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

    def process_images(self, test_images_dir: str) -> list:
        """
        Метод process_images() обрабатывает все изображения в указанной директории,
        выполняя инференс для каждого изображения.
        Parameters:
            test_images_dir (str): Путь к директории с изображениями
        Returns:
            inference_results (list): Список словарей с именами файлов изображений,
                масками и метками
        """
        if not os.path.isdir(test_images_dir):
            self.logger.error(f"{test_images_dir} is not a directory")
            raise NotADirectoryError(f"{test_images_dir} is not a directory")

        test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if
                       f.endswith(('.jpg', '.png'))]
        inference_results = []
        for image_path in test_images:
            results = self.run_inference(image_path)
            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()
                filename = os.path.split(image_path)[-1]
                img_result = {"filename": filename, "masks": masks, "labels": labels}
                inference_results.append(img_result)

            else:
                self.logger.warning(f"No objects in {image_path}")

        return inference_results