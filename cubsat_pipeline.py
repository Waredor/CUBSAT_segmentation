import os

from utils.config_manager import ConfigManager, setup_logger
from utils.model_trainer import ModelTrainer
from utils.model_exporter import ModelExporter
from utils.inference_runner import InferenceRunner
from utils.annotation_processor import AnnotationProcessor


if __name__ == '__main__':

    ########################################################
    ########     HYPERPARAMETERS AND CONFIGS        ########
    ########################################################

    init_path = os.path.abspath(__file__)

    def get_project_root(start_path):
        current = start_path
        while current != os.path.dirname(current):
            if os.path.exists(os.path.join(current, "requirements.txt")):
                return current
            current = os.path.dirname(current)
        raise FileNotFoundError("Project root was not found")

    project_root_path = get_project_root(init_path)

    # Logger init
    CUBSAT_LOG_FILE = os.path.join(project_root_path, "src", "logs", "cubsat_log.txt")
    CUBSAT_LOGGER_NAME = "cubsat_pipeline_logger"
    LOGGER = setup_logger(
        logger_name=CUBSAT_LOGGER_NAME,
        logger_file_path=CUBSAT_LOG_FILE
    )

    # Путь к корневой папке с датасетом (изменить на свой)
    #DATA_ROOT_PATH = "D:\\Python projects\\CUBSAT_Dataset_segmentation\\Fine_tuning"
    DATA_ROOT_PATH = "C:\\Python projects\\Datasets\\Fine_tuning"

    # Параметры модели для обучения
    MODEL_HYPERPARAMETERS = os.path.join(
        project_root_path,
        "configs",
        "model_cfg.json"
    )

    # Конфигурационный файл датасета
    DATA_CFG = os.path.join(
        DATA_ROOT_PATH,
        "dataset.yaml"
    )

    # Файл с предобученной моделью
    MODEL_PATH = os.path.join(
        project_root_path,
        "models",
        "yolo11n-seg.pt"
    )

    # Директория для сохранения модели
    OUTPUT_DIR = os.path.join(
        project_root_path,
        "models"
    )

    # Флаг для создания аннотаций
    CREATE_ANNOTATIONS = False

    # Поперечный размер изображений в датасете
    IMG_SIZE = 1024

    ########################################################
    ###############     CHECK CONFIGS        ###############
    ########################################################

    config_manager = ConfigManager(
        data_cfg=DATA_CFG,
        model_hyperparameters=MODEL_HYPERPARAMETERS,
        data_dir=DATA_ROOT_PATH,
        model_cfg=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        logger = LOGGER
    )

    # Валидация конфигурационных файлов модели
    try:
        config = config_manager.load_config()

    except Exception as e:
        LOGGER.error(f"Ошибка валидации конфигурационных файлов: {str(e)}")
        raise

    ########################################################
    ################     TRAIN MODEL        ################
    ########################################################

    model_trainer = ModelTrainer(
        model_cfg=MODEL_PATH,
        hyperparameters=config,
        logger=LOGGER
    )

    # Обучение модели
    try:
        model_trainer.train_model()

    except Exception as e:
        LOGGER.error(f"Ошибка обучения модели: {str(e)}")
        raise


    ########################################################
    ################     SAVE MODEL        #################
    ########################################################

    model_exporter = ModelExporter(
        model=model_trainer.model,
        output_dir=OUTPUT_DIR,
        logger=LOGGER
    )

    # Сохранение обученной модели
    try:
        model_exporter.save_model("yolo11n-seg_fine_tuned.pt")

    except Exception as e:
        LOGGER.error(f"Ошибка охранения модели: {str(e)}")
        raise


    ########################################################
    #############    CREATE ANNOTATIONS        #############
    ########################################################

    if CREATE_ANNOTATIONS is True:
        inference_runner = InferenceRunner(
            model=model_trainer.model,
            img_size=IMG_SIZE,
            logger=LOGGER
        )

        # Директория с изображениями для инференса
        TEST_DIR = os.path.join(DATA_ROOT_PATH, "images", "test")

        # Директория для сохранения аннотаций
        ANNOTATIONS_DIR = os.path.join(TEST_DIR, "annotations")

        # Запуск инференса
        try:
            inference_results = inference_runner.process_images(TEST_DIR)

        except Exception as e:
            LOGGER.error(f"Ошибка инференса: {str(e)}")
            raise

        annotation_processor = AnnotationProcessor(
            class_names=config["class_names"],
            logger=LOGGER
        )

        # Создание аннотаций в формате LabelMe
        for result in inference_results:
            image_path = os.path.join(TEST_DIR, result["filename"])
            try:
                annotation_processor.create_labelme_json(
                    image_path=image_path,
                    masks=result["masks"],
                    labels=result["labels"],
                    output_dir=ANNOTATIONS_DIR
                )

            except Exception as e:
                LOGGER.error(f"Ошибка создания аннотаций: {str(e)}")
                raise