import logging
import ultralytics

from typing import Any
from ultralytics import YOLO
from torch import cuda


def train_model(
    model: ultralytics.models.yolo.model.YOLO, logger: logging.Logger, hyperparameters: dict, augment=False
) -> tuple[YOLO, Any]:
    """
    Метод train_model() выполняет обучение модели YOLOv11.
    Parameters:
        model (ultralytics.models.yolo.model.YOLO): инициализированная модель YOLOv11.
        hyperparameters (dict): словарь с гиперпараметрами модели для обучения
            и путем к конфигурационному файлу датасета.
        logger (logging.Logger): объект логгера.
        augment (bool): флаг использования аугментаций.
    Returns:
        tuple[model (ultralytics.models.yolo.model.YOLO), results (ultralytics.engine.trainer.TrainResults)]
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
    final_learning_rate = hyperparameters['lrf']
    optimizer = hyperparameters['optimizer']
    patience = hyperparameters['patience']
    device = hyperparameters['device']
    dropout = hyperparameters['dropout']
    label_smoothing = hyperparameters['label_smoothing']
    warmup_epochs = hyperparameters['warmup_epochs']
    iou = hyperparameters['iou']
    weight_decay = hyperparameters['weight_decay']
    cos_lr = hyperparameters['cos_lr']
    workers = hyperparameters['num_workers']

    if device == 0:
        if cuda.is_available():
            logger.info("Using GPU device")

        else:
            logger.info("Using CPU device")
            device = "cpu"

    augment_params = hyperparameters["augment_params"] if augment else {}

    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        lr0=initial_learning_rate,
        optimizer=optimizer,
        patience=patience,
        device=device,
        freeze=num_layers_to_freeze,
        workers=workers,
        project="runs/train",
        name="exp",
        exist_ok=True,
        cos_lr=cos_lr,
        lrf=final_learning_rate,
        dropout=dropout,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        warmup_epochs=warmup_epochs,
        iou=iou,
        cache=False,
        split="val",
        **augment_params
    )

    logger.info("Training completed")
    return model, results