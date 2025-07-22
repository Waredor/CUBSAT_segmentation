import os
from utils import Pipeline


if __name__ == '__main__':
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    while not os.path.exists(os.path.join(current_dir, '.venv')):
        current_dir = os.path.dirname(current_dir)
        if current_dir == os.path.dirname(current_dir):
            raise FileNotFoundError("Папка .venv не найдена в проекте")

    project_root_path = os.path.abspath(current_dir)

    # Путь к корневой папке с датасетом (изменить на свой)
    DATA_ROOT_PATH = str(project_root_path) + "\\src\\tests\\test_data"
    #DATA_ROOT_PATH = "D:\\Python projects\\CUBSAT_Dataset_segmentation\\Fine_tuning"

    # Hyperparameters and configs
    MODEL_CFG = str(project_root_path) + "\\Model_cfg\\model_cfg.json"      # параметры модели для обучения
    DATA_CFG = DATA_ROOT_PATH + "\\valid_dataset.yaml"      # конфигурация датасета
    MODEL_PATH = str(project_root_path) + "\\Model_cfg\\yolo11n-seg_labeling.pt"        # файл с предобученной моделью
    OUTPUT_DIR = str(project_root_path) + "\\Model_cfg\\"

    #Эти пути нужно изменить на свои
    OUTPUT_PATH_IMG = DATA_ROOT_PATH + "\\images\\test"
    OUTPUT_PATH_LABELS = DATA_ROOT_PATH + "\\labels\\test"
    YOLO_CONVERT_LABELS_PATH = DATA_ROOT_PATH + "\\labels\\train"
    LABELME_INPUT_LABELS_PATH = DATA_ROOT_PATH + "\\labels\\json_labels\\"

    # Pipeline init
    labeling_pipeline = Pipeline(
        model_hyperparameters=MODEL_CFG,
        data_cfg=DATA_CFG,
        data_dir=DATA_ROOT_PATH,
        output_dir=OUTPUT_DIR,
        model_cfg=MODEL_PATH,
    )

    #Run Pipeline
    labeling_pipeline.fine_tune_for_labeling(model_filename='yolo11n-seg_labeling.pt')
    labeling_pipeline.create_new_json_annotations(
        test_images_dir=OUTPUT_PATH_IMG,
        annotations_output_dir=OUTPUT_PATH_LABELS,
    )

    #labeling_pipeline.convert_labelme_to_yolo(
        #labelme_annotations_path=labelme_input_labels_path,
        #yolo_annotations_path=yolo_convert_labels_path
    #)