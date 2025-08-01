import logging
import os
import shutil
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataConfigurator:
    """
    Класс Data Configurator отвечает за конфигурацию датасета
    перед обучением модели.
    Parameters:
        source_dir (str): корневая директория с необработанными данными
        destination_dir (str): корневая директория датасета для обучения модели
        images_extensions (list): список расширений файлов изображений
        labels_extensions (list): список расширений файлов аннотаций
        logger (logging.Logger): объект логгера
    """
    def __init__(
            self,
            source_dir: str,
            destination_dir: str,
            images_extensions: list,
            labels_extensions: list,
            logger: logging.Logger
    ) -> None:
        self.source_dir = source_dir
        self.destination_dir = destination_dir
        self.images_extensions = images_extensions
        self.labels_extensions = labels_extensions
        self.logger = logger
        self.target_filenames = []
        self.images_filenames = []
        self.path_to_images = os.path.join(self.source_dir, "images")
        self.path_to_labels = os.path.join(self.source_dir, "labels")

    def process_image_file(self, el) -> None:
        full_path = os.path.join(self.path_to_images, el)
        if os.path.isfile(full_path):
            if Path(el).suffix in self.images_extensions:
                self.images_filenames.append(el)

        return None

    def process_label_file(self, el) -> None:
        full_path = os.path.join(self.path_to_labels, el)
        if os.path.isfile(full_path):
            if Path(el).suffix in self.labels_extensions:
                self.target_filenames.append(el)

        return None

    def train_test_split(self, max_workers: int) -> None:
        """
        Метод train_test_split() осуществляет разделение данных
        на обучающую и тестовую/валидационную выборки, а
        также перенос аннотаций и изображений в соответствующие папки в корневой директории
        датасета.
        Parameters:
            max_workers (int): количество потоков для параллельной обработки данных
        Returns:
            None
        """
        if not os.path.exists(self.path_to_labels):
            self.logger.error("Labels directory does not exist")
            raise NotADirectoryError("Labels directory does not exist")

        if not os.path.exists(self.path_to_images):
            self.logger.error("Images directory does not exist")
            raise NotADirectoryError("Images directory does not exist")

        self.logger.info("Starting creating dataframes...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_image_file, os.listdir(self.path_to_images))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.process_label_file, os.listdir(self.path_to_labels))

        data_x = {'images': self.images_filenames}
        data_y = {'labels': self.target_filenames}
        dataset_x = pd.DataFrame(data=data_x)
        dataset_y = pd.DataFrame(data=data_y)

        self.logger.info("Dataframes created")
        self.logger.info("Starting train-test split...")

        x_train, x_test, y_train, y_test = train_test_split(
            dataset_x,
            dataset_y,
            train_size=0.85,
            random_state=42,
            shuffle=True
        )

        x_train["type"] = "train"
        x_test["type"] = "val"
        y_train["type"] = "train"
        y_test["type"] = "val"

        x_dataset = pd.concat([x_train, x_test], axis=0)
        y_dataset = pd.concat([y_train, y_test], axis=0)

        self.logger.info("Train-test split done")
        self.logger.info("Starting moving files into dataset dir...")

        for index, row in x_dataset.iterrows():
            filename = row["images"]
            destination_dir = row["type"]

            source_path = os.path.join(self.source_dir, "images", filename)
            destination_path = os.path.join(self.destination_dir, "images", destination_dir)

            if os.path.exists(source_path):
                try:
                    if (destination_dir == "val" and filename
                            not in os.listdir(os.path.join(self.destination_dir, "images",
                                                           "train"))):
                        shutil.copy(source_path, destination_path)
                        self.logger.info(f"Moving {filename} to {destination_path}")

                    elif destination_dir == "train":
                        shutil.copy(source_path, destination_path)
                        self.logger.info(f"Moving {filename} to {destination_path}")

                except Exception as e:
                    print(e)

        for index, row in y_dataset.iterrows():
            filename = row["labels"]
            destination_dir = row["type"]

            source_path = os.path.join(self.source_dir, "labels", filename)
            destination_path = os.path.join(self.destination_dir, "labels", destination_dir)

            if os.path.exists(source_path):
                try:
                    if (destination_dir == "val" and filename
                            not in os.listdir(os.path.join(self.destination_dir, "labels",
                                                           "train"))):
                        shutil.copy(source_path, destination_path)
                        self.logger.info(f"Moving {filename} to {destination_path}")

                    elif destination_dir == "train":
                        shutil.copy(source_path, destination_path)
                        self.logger.info(f"Moving {filename} to {destination_path}")

                except Exception as e:
                    print(e)

        self.logger.info("Configuration successfully completed")