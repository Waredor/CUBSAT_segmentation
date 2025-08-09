import logging
import logging.handlers
import unittest
import os
import numpy as np

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
            "models",
            "model.pt"
        )
        self.model = YOLO(model_path)
        self.img_size = 1024
        self.temp_dir = os.path.join(
            project_root_path,
            "utils",
            "tests",
            "test_data"
        )

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)
        self.log_handler.close()

    def test_run_inference_success(self):
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
        results = inference_runner.run_inference(
            image_path, batch_size=1, confidence=0.5, iou=0.7
        )
        self.assertTrue(len(results) > 0)

    def test_run_inference_file_not_found_error(self):
        """
        Этот тест осуществляет проверку работоспособности метода
        InferenceRunner.run_inference() при неверном пути к файлу изображения.
        """
        inference_runner = InferenceRunner(
            model=self.model,
            img_size=self.img_size,
            logger=self.logger
        )

        wrong_image_path = os.path.join(
            project_root_path,
            "utils",
            "tests",
            "test_data",
            "images",
            "train",
            "0005.jpg"
        )

        with self.assertRaises(FileNotFoundError) as cm:
            inference_runner.run_inference(
                wrong_image_path, batch_size=1, confidence=0.5, iou=0.7
            )
            self.assertEqual(
                first=str(cm.exception),
                second=f"File {wrong_image_path} doesn't found"
            )

    def test_process_images_success(self):
        """
        Этот тест осуществляет проверку работоспособности метода
        InferenceRunner.process_images() при валидных входнх данных.
        """
        inference_runner = InferenceRunner(
            model=self.model,
            img_size=self.img_size,
            logger=self.logger
        )

        image_dir = os.path.join(
            project_root_path,
            "utils",
            "tests",
            "test_data",
            "images",
            "train"
        )

        results = inference_runner.process_images(
            image_dir, batch_size=1, confidence=0.5, iou=0.7
        )
        self.assertEqual(type(results), list)
        for result in results:
            self.assertEqual(type(result), dict)
            self.assertTrue("masks" in result.keys())
            self.assertTrue("filename" in result.keys())
            self.assertTrue("labels" in result.keys())
            for key, value in result.items():
                if key == "filename":
                    self.assertEqual(type(value), str)

                elif key == "masks" or key == "labels":
                    self.assertEqual(type(value), np.ndarray)

    def test_process_images_not_a_directory_error(self):
        """
        Этот тест осуществляет проверку работоспособности метода
        InferenceRunner.process_images() при неверном пути к директории с изображениями.
        """
        inference_runner = InferenceRunner(
            model=self.model,
            img_size=self.img_size,
            logger=self.logger
        )

        wrong_image_dir = os.path.join(
            project_root_path,
            "utils",
            "abc",
            "wrong_dir"
        )

        with self.assertRaises(NotADirectoryError) as cm:
            inference_runner.process_images(
                wrong_image_dir, batch_size=1, confidence=0.5, iou=0.7
            )
            self.assertEqual(
                first=str(cm.exception),
                second=f"{wrong_image_dir} is not a directory"
            )