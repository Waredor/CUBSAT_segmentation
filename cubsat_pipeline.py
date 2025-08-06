import os
import yaml

from ultralytics import YOLO
from utils.config_manager import ConfigManager, setup_logger
from utils.model_trainer import train_model
from utils.inference_runner import InferenceRunner
from utils.annotation_processor import AnnotationProcessor
from utils.data_configurator import DataConfigurator


if __name__ == '__main__':

    ########################################################
    ########     HYPERPARAMETERS AND CONFIGS        ########
    ########################################################

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    init_path = os.path.abspath(__file__)

    def get_project_root(start_path):
        current = start_path
        while current != os.path.dirname(current):
            if os.path.exists(os.path.join(current, "requirements.txt")):
                return current
            current = os.path.dirname(current)
        raise FileNotFoundError("Project root was not found")

    project_root_path = get_project_root(init_path)


    CUBSAT_LOG_FILE = os.path.join(
        project_root_path, "logs", "cubsat_pipeline_log.txt"
    )
    CUBSAT_LOGGER_NAME = "cubsat_pipeline_logger"
    LOGGER = setup_logger(
        logger_name=CUBSAT_LOGGER_NAME,
        logger_file_path=CUBSAT_LOG_FILE
    )


    pipeline_config_path = os.path.join(
        project_root_path, "configs", "pipeline_config.yaml"
    )
    with open(pipeline_config_path, mode='r', encoding='utf-8') as f:
        pipeline_yaml_config = yaml.safe_load(f)


    ########################################################
    ###############     CHECK CONFIGS        ###############
    ########################################################

    config_manager = ConfigManager(
        project_root=project_root_path,
        pipeline_cfg=pipeline_config_path,
        logger = LOGGER
    )

    try:
        config = config_manager.load_config()

        stages = pipeline_yaml_config["stages"]

        MODEL_HYPERPARAMETERS = pipeline_yaml_config["model_hyperparameters"]
        IMG_SIZE = MODEL_HYPERPARAMETERS["imgsz"]

        MODEL_PATH = os.path.join(
            project_root_path, "models", str(pipeline_yaml_config["names"]["pt_model_name"])
        )
        EXPORT_MODEL_PATH = os.path.join(
            project_root_path, "models", str(pipeline_yaml_config["names"]["exported_model_name"])
        )

        IMAGES_EXTENSIONS = pipeline_yaml_config["extensions"]["images_extensions"]
        LABELS_EXTENSIONS = pipeline_yaml_config["extensions"]["labels_extensions"]

        DATASET_CFG = pipeline_yaml_config["dataset_cfg"]
        DATA_ROOT_PATH = DATASET_CFG["path"]
        RAW_DATA_DIR = pipeline_yaml_config["paths"]["raw_data_dir"]

        model = YOLO(MODEL_PATH)
        LOGGER.info("Модель успешно инициализирована")

    except Exception as e:
        LOGGER.error(f"Ошибка валидации конфигурационных файлов: {str(e)}")
        raise


    ########################################################
    ##############     PIPELINE STAGES        ##############
    ########################################################

    for stage in stages:
        if stage == "configure_data":
            data_configurator = DataConfigurator(
                source_dir=RAW_DATA_DIR,
                destination_dir=DATA_ROOT_PATH,
                images_extensions=IMAGES_EXTENSIONS,
                labels_extensions=LABELS_EXTENSIONS,
                logger=LOGGER
            )
            try:
                data_configurator.train_test_split(max_workers=3)

            except Exception as e:
                LOGGER.error(f"Ошибка предобработки данных: {str(e)}")
                raise


        elif stage == "train_model":
            try:
                model = train_model(model=model, hyperparameters=MODEL_HYPERPARAMETERS, logger=LOGGER, augment=True)

            except Exception as e:
                LOGGER.error(f"Ошибка обучения модели: {str(e)}")
                raise


        elif stage == "export_model":
            try:
                model.save(EXPORT_MODEL_PATH)

            except Exception as e:
                LOGGER.error(f"Ошибка охранения модели: {str(e)}")
                raise


        elif stage in ("create_labelme_annotations", "create_yolo_annotations"):
            INFERENCE_IMAGES_DIR = str(pipeline_yaml_config["paths"]["inference_images_dir"])
            LABELME_ANNOTATIONS_DIR = str(pipeline_yaml_config["paths"]["inference_annotations_dir"])
            YOLO_ANNOTATIONS_DIR = os.path.join(RAW_DATA_DIR, "labels")

            annotation_processor = AnnotationProcessor(
                class_names=config["class_names"],
                yolo_annotations_path=YOLO_ANNOTATIONS_DIR,
                labelme_annotations_path=LABELME_ANNOTATIONS_DIR,
                logger=LOGGER
            )


            if stage == "create_labelme_annotations":
                inference_runner = InferenceRunner(
                    model=model,
                    img_size=IMG_SIZE,
                    logger=LOGGER
                )
                try:
                    inference_results = inference_runner.process_images(
                        INFERENCE_IMAGES_DIR,
                        batch_size=config["batch"],
                    )

                except Exception as e:
                    LOGGER.error(f"Ошибка инференса: {str(e)}")
                    raise

                for result in inference_results:
                    image_path = os.path.join(INFERENCE_IMAGES_DIR, result["filename"])
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

            elif stage == "create_yolo_annotations":
                try:
                    annotation_processor.convert_labelme_to_yolo()

                except Exception as e:
                    LOGGER.error(f"Ошибка создания аннотаций: {str(e)}")
                    raise