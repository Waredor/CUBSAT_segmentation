from utils import Pipeline

# Hyperparameters and configs
model_cfg = "Model_cfg/model_cfg.json" # параметры модели для обучения
data_cfg = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/dataset.yaml" # конфигурация датасета
data_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/"
output_path_img = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/images/test/"
output_path_labels = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/labels/test/"
yolo_convert_labels_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/labels/train/"
labelme_input_labels_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/test_output/label_me_json/"
model_path = "../Model_cfg/yolo11n-seg.pt"
model_output_path = "../Model_cfg/"
class_names = ["FT", "Engine", "Solar Panel"]


# Pipeline init
labeling_pipeline = Pipeline(
    model_hyperparameters=model_cfg,
    data_cfg=data_cfg,
    data_dir=data_path,
    output_dir=output_path_img,
    model_cfg=model_path,
)

#Run Pipeline
labeling_pipeline.fine_tune_for_labeling(model_output_dir=model_output_path)
labeling_pipeline.create_new_json_annotations(
    test_images_dir=output_path_img,
    annotations_output_dir=output_path_labels
)

#labeling_pipeline.convert_labelme_to_yolo(
    #labelme_annotations_path=labelme_input_labels_path,
    #yolo_annotations_path=yolo_convert_labels_path,
#)