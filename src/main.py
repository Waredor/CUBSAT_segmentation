from utils import Pipeline


if __name__ == '__main__':
    # Hyperparameters and configs
    model_cfg = "D:/Python projects/CUBSAT_segmentation/Model_cfg/model_cfg.json" # параметры модели для обучения
    data_cfg = "D:/Python projects/CUBSAT_Dataset_segmentation/Fine_tuning/dataset.yaml" # конфигурация датасета
    data_path = "D:/Python projects/CUBSAT_Dataset_segmentation/Fine_tuning/"
    output_path_img = "D:/Python projects/CUBSAT_Dataset_segmentation/Fine_tuning/images/test/"
    output_path_labels = "D:/Python projects/CUBSAT_Dataset_segmentation/Fine_tuning/labels/test/"
    yolo_convert_labels_path = "D:/Python projects/CUBSAT_Dataset_segmentation/Fine_tuning/labels/train/"
    labelme_input_labels_path = "C:/Users/Екатерина/Desktop/ML ЦНИХМ/Проекты/Datasets/test_output/label_me_json/"
    model_path = "D:/Python projects/CUBSAT_segmentation/Model_cfg/yolo11n-seg_labeling.pt"
    model_output_path = "D:/Python projects/CUBSAT_segmentation/Model_cfg/yolo11n-seg_labeling.pt"
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
        #yolo_annotations_path=yolo_convert_labels_path
    #)