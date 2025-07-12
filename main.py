from utils import Pipeline
from ultralytics import YOLO

# Hyperparameters and configs
model_cfg = "Model_cfg/model_cfg.json" # параметры модели для обучения
data_cfg = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/dataset.yaml" # конфигурация датасета
data_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/"
output_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/Fine_tuning/images/test/"
model_path = "Model_cfg/yolo11n-seg.pt"
model_output_path = "Model_cfg/"
class_names = ["FT", "Engine", "Solar Panel"]

#model = YOLO(model_path)
#modules = list(model.model.modules())
#for module in modules:
    #print(module)



labeling_pipeline = Pipeline(
    model_hyperparameters=model_cfg,
    data_cfg=data_cfg,
    data_dir=data_path,
    output_dir=output_path,
    model_cfg=model_path,
)

labeling_pipeline.fine_tune_for_labeling(model_output_path)