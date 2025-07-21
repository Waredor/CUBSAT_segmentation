import pytest
import tempfile
import sys
import os
import yaml
import json
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src import utils

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def sample_image(temp_dir):
    image_path = os.path.join(temp_dir, "sample_image.jpg")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(image_path, img)
    yield image_path


@pytest.fixture
def valid_config_files(temp_dir):
    dataset_yaml = os.path.join(temp_dir, "dataset.yaml")
    hyperparameters_json = os.path.join(temp_dir, "hyperparameters.json")
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"path": temp_dir, "train": "images/train", "val": "images/val",
                   "nc": 4, "names": ["FT", "Engine", "Solar Panel", "background"]}, f)
    with open(hyperparameters_json, "w", encoding="utf-8") as f:
        json.dump({"epochs": 10, "imgsz": 640, "batch": 16, "lr0": 0.01,
                   "optimizer": "Adam", "patience": 5, "device": "cpu",
                   "freeze_layers": 10}, f)
    yield dataset_yaml, hyperparameters_json


class TestConfigManager:
    def test_validate_config_success(self, temp_dir, valid_config_files, mocker):
        dataset_yaml, hyperparameters_json = valid_config_files
        os.makedirs(os.path.join(temp_dir, "images/train"))
        os.makedirs(os.path.join(temp_dir, "images/val"))
        config_manager = utils.ConfigManager(
            data_cfg=dataset_yaml,
            model_hyperparameters=hyperparameters_json,
            data_dir=temp_dir,
            model_cfg=os.path.join(temp_dir, "model.pt"),
            output_dir=temp_dir
        )
        mocker.patch.object(config_manager.logger, "info")
        config_manager.validate_config()
        assert config_manager.logger.info.call_count == 2
        config_manager.logger.info.assert_any_call("Начало валидации")
        config_manager.logger.info.assert_any_call("Валидация завершена")

    def test_validate_config_none_data_cfg(self, temp_dir, mocker):
        config_manager = utils.ConfigManager(
            data_cfg=None,
            model_hyperparameters=os.path.join(temp_dir, "hyperparameters.json"),
            data_dir=temp_dir,
            model_cfg=os.path.join(temp_dir, "model.pt"),
            output_dir=temp_dir
        )
        mocker.patch.object(config_manager.logger, "error")
        with pytest.raises(ValueError) as exc_info:
            config_manager.validate_config()
        assert "None" in str(exc_info.value)
        config_manager.logger.error.assert_called_once()
