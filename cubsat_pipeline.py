import os
import yaml

from utils.config_manager import ConfigManager, setup_logger
from utils.model_trainer import ModelTrainer
from utils.model_exporter import ModelExporter
from utils.inference_runner import InferenceRunner
from utils.annotation_processor import AnnotationProcessor
from utils.data_configurator import DataConfigurator


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

    # Инициализация логгера
    CUBSAT_LOG_FILE = os.path.join(project_root_path, "logs", "cubsat_log.txt")
    CUBSAT_LOGGER_NAME = "cubsat_pipeline_logger"
    LOGGER = setup_logger(
        logger_name=CUBSAT_LOGGER_NAME,
        logger_file_path=CUBSAT_LOG_FILE
    )

    # Загрузка конфигурационного файла пайплайна
    pipeline_config_path = os.path.join(
        project_root_path, "configs", "pipeline_config.yaml"
    )
    with open(pipeline_config_path, mode='r', encoding='utf-8') as f:
        pipeline_yaml_config = yaml.safe_load(f)

    # Путь к корневой папке с датасетом
    DATA_ROOT_PATH = pipeline_yaml_config["data_root_path"]

    # Флаг для предобработки данных
    CONFIGURE_DATA = pipeline_yaml_config["configure_data"]

    # Флаг для обучения модели
    TRAIN_MODEL = pipeline_yaml_config["train_model"]

    # Флаг для создания аннотаций
    CREATE_ANNOTATIONS_FLAG = pipeline_yaml_config["create_annotations"]["flag"]

    # Флаг для запуска AnnotationProcessor.create_labelme_json()
    CREATE_ANNOTATIONS_LABELME = pipeline_yaml_config["create_annotations"]["labelme"]

    # Флаг для запуска AnnotationProcessor.convert_labelme_to_yolo()
    CREATE_ANNOTATIONS_YOLO = pipeline_yaml_config["create_annotations"]["yolo"]

    # Список допустимых расширений файлов изображений
    IMAGES_EXTENSIONS = pipeline_yaml_config["images_extensions"]

    # Список допустимых расширений файлов изображений
    LABELS_EXTENSIONS = pipeline_yaml_config["labels_extensions"]

    # Директория с необработанными данными
    RAW_DATA_DIR = pipeline_yaml_config["raw_data_dir"]

    # Параметры модели для обучения
    MODEL_HYPERPARAMETERS = os.path.join(project_root_path, "configs", "model_cfg.json")

    # Конфигурационный файл датасета
    DATA_CFG = os.path.join(DATA_ROOT_PATH, "dataset.yaml")

    # Файл с предобученной моделью
    MODEL_PATH = os.path.join(project_root_path, "models", "yolo11n-seg.pt")

    # Директория для сохранения модели
    OUTPUT_DIR = os.path.join(project_root_path, "models")

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
    ##############     CONFIGURE DATA        ###############
    ########################################################

    if CONFIGURE_DATA:
        data_configurator = DataConfigurator(
            source_dir=RAW_DATA_DIR,
            destination_dir=DATA_ROOT_PATH,
            images_extensions=IMAGES_EXTENSIONS,
            labels_extensions=LABELS_EXTENSIONS,
            logger=LOGGER
        )

        # Предобработка данных
        try:
            data_configurator.train_test_split(max_workers=3)

        except Exception as e:
            LOGGER.error(f"Ошибка предобработки данных: {str(e)}")
            raise


    ########################################################
    ################     TRAIN MODEL        ################
    ########################################################

    model_trainer = ModelTrainer(
        model_cfg=MODEL_PATH,
        hyperparameters=config,
        logger=LOGGER
    )

    if TRAIN_MODEL:
        # Обучение модели
        try:
            model_trainer.train_model()

        except Exception as e:
            LOGGER.error(f"Ошибка обучения модели: {str(e)}")
            raise


    ########################################################
    ################     SAVE MODEL        #################
    ########################################################

    if TRAIN_MODEL:
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

    if CREATE_ANNOTATIONS_FLAG:
        # Директория с изображениями для инференса
        TEST_DIR = os.path.join(DATA_ROOT_PATH, "images", "test")

        # Директория с аннотациями в формате LabelMe
        LABELME_ANNOTATIONS_DIR = os.path.join(DATA_ROOT_PATH, "labels", "test")

        # Директория для сохранения аннотаций в формате YOLO
        YOLO_ANNOTATIONS_DIR = os.path.join(RAW_DATA_DIR, "labels")

        annotation_processor = AnnotationProcessor(
            class_names=config["class_names"],
            yolo_annotations_path=YOLO_ANNOTATIONS_DIR,
            labelme_annotations_path=LABELME_ANNOTATIONS_DIR,
            logger=LOGGER
        )

        if CREATE_ANNOTATIONS_LABELME:
            inference_runner = InferenceRunner(
                model=model_trainer.model,
                img_size=IMG_SIZE,
                logger=LOGGER
            )

            # Запуск инференса
            try:
                inference_results = inference_runner.process_images(
                    TEST_DIR,
                    batch_size=config["batch"],
                )

            except Exception as e:
                LOGGER.error(f"Ошибка инференса: {str(e)}")
                raise

            # Создание аннотаций в формате LabelMe
            for result in inference_results:
                image_path = os.path.join(TEST_DIR, result["filename"])
                try:
                    annotation_processor.create_labelme_json(
                        image_path=image_path,
                        masks=result["masks"],
                        labels=result["labels"],
                        output_dir=LABELME_ANNOTATIONS_DIR
                    )

                except Exception as e:
                    LOGGER.error(f"Ошибка создания аннотаций: {str(e)}")
                    raise

        if CREATE_ANNOTATIONS_YOLO:
            # Создание аннотаций в формате YOLO
            try:
                annotation_processor.convert_labelme_to_yolo()

            except Exception as e:
                LOGGER.error(f"Ошибка создания аннотаций: {str(e)}")
                raise