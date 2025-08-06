import os
import logging
import ultralytics

from ultralytics import YOLO
from torch import cuda


def train_model(
    model: ultralytics.models.yolo.model.YOLO, logger: logging.Logger, hyperparameters: dict, augment=False
) -> YOLO:
    """
    Метод train_model() выполняет обучение модели YOLOv11.
    Parameters:
        model (ultralytics.models.yolo.model.YOLO): инициализированная модель YOLOv11.
        hyperparameters (dict): словарь с гиперпараметрами модели для обучения
            и путем к конфигурационному файлу датасета.
        logger (logging.Logger): объект логгера.
        augment (bool): флаг использования аугментаций.
    Returns:
        model (ultralytics.models.yolo.model.YOLO)
    """
    logger.info("Starting training")
    logger.info(f"Training model with parameters: {hyperparameters}")
    cuda.empty_cache()
    num_layers_to_freeze = hyperparameters['freeze_layers']
    data_path = hyperparameters['data_path']
    epochs = hyperparameters['epochs']
    batch_size = hyperparameters['batch']
    image_size = hyperparameters['imgsz']
    initial_learning_rate = hyperparameters['lr0']
    optimizer = hyperparameters['optimizer']
    patience = hyperparameters['patience']
    device = hyperparameters['device']

    if device == 0:
        if cuda.is_available():
            logger.info("Using GPU device")

        else:
            logger.info("Using CPU device")
            device = "cpu"

    augment_params = hyperparameters["augment_params"] if augment else {}

    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        lr0=initial_learning_rate,
        optimizer=optimizer,
        patience=patience,
        device=device,
        freeze=num_layers_to_freeze,
        workers=2,
        project="runs/train",
        name="exp",
        exist_ok=True,
        cos_lr=True,
        lrf=0.005,
        dropout=0.3,
        weight_decay=0.001,
        label_smoothing=0.1,
        warmup_epochs=3,
        iou=0.7,
        **augment_params
    )

    logger.info("Training completed")
    return model