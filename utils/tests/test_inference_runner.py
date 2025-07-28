import logging
import logging.handlers
import unittest
import os

from utils.inference_runner import InferenceRunner
from ultralytics import YOLO

def get_project_root():
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    return os.path.abspath(os.path.join(current_dir, '..', '..'))

project_root_path = get_project_root()

class TestInferenceRunner(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_config_manager_logger")
        self.logger.setLevel(logging.INFO)

        self.log_handler = logging.handlers.MemoryHandler(capacity=1000)
        self.logger.addHandler(self.log_handler)

        model_path = os.path.join(
            project_root_path,
            "utils",
            "tests",
            "test_data",
            "model.pt"
        )
        self.model = YOLO(model_path)
        self.img_size = 1024

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()

    def run_inference_success(self):
        """
        Этот тест осуществляет проверку работоспособности метода
        InferenceRunner.run_inference() при валидных входных данных.
        """
        inference_runner = InferenceRunner(
            model=self.model,
            img_size=self.img_size,
            logger=self.logger
        )

        image_path = os.path.join(
            project_root_path,
            "utils",
            "tests",
            "test_data",
            "images",
            "train",
            "0001.jpg"
        )
        results = inference_runner.run_inference(image_path)
        self.assertTrue(len(results) > 0)