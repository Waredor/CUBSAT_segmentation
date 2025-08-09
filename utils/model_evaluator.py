import logging
import ultralytics

from torch import cuda


def evaluate(
    model: ultralytics.models.yolo.model.YOLO, logger: logging.Logger, hyperparameters: dict
) -> None:
    """
    Метод evaluate() отвечает за валидацию обученной модели YOLOv11
    на валидационной выборке.
    Parameters:
        model (ultralytics.models.yolo.model.YOLO): модель YOLOv11.
        logger (logging.Logger): объект логгера.
        hyperparameters (dict): словарь с гиперпараметрами модели.
    """
    logger.info("Starting evaluation")
    cuda.empty_cache()

    data_path = hyperparameters['data_path']
    image_size = hyperparameters['imgsz']
    iou = hyperparameters['iou']
    device = hyperparameters['device']
    batch = hyperparameters['batch']

    if device == 0:
        if cuda.is_available():
            logger.info("Using GPU device")

        else:
            logger.info("Using CPU device")
            device = "cpu"

    metrics = model.val(
        data=data_path,
        imgsz=image_size,
        iou=iou,
        device=device,
        batch=batch,
        conf=0.8,
        save=True,
        split="test"
    )
    logger.info("Evaluation finished")
    logger.info(f"Boxes mAP50-95: {metrics.box.map}")
    logger.info(f"Boxes mAP50: {metrics.box.map50}")
    logger.info(f"Boxes precision: {metrics.box.mp}")
    logger.info(f"Boxes recall: {metrics.box.mr}")

    logger.info(f"Masks mAP50-95: {metrics.seg.map}")
    logger.info(f"Masks mAP50: {metrics.seg.map50}")
    logger.info(f"Masks precision: {metrics.seg.mp}")
    logger.info(f"Masks recall: {metrics.seg.mr}")