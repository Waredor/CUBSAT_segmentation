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
        self.path_to_images = os.path.join(self.source_dir, "images")
        self.path_to_labels = os.path.join(self.source_dir, "labels")

    def move_train_image(self, filename: str) -> tuple[str, bool, str]:
        """
        Метод move_train_image() копирует изображение в папку images/train.
        Parameters:
            filename (str): Имя файла изображения
        Returns:
            tuple: (имя файла, успех копирования, сообщение об ошибке если есть)
        """
        source_path = os.path.join(self.path_to_images, filename)
        destination_path = os.path.join(self.destination_dir, "images", "train", filename)

        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)
                if os.path.exists(destination_path):
                    return filename, True, f"Copied {filename} to {destination_path}"
                else:
                    return filename, False, f"Failed to copy {filename}: destination file not created"
            else:
                return filename, False, f"Source file {source_path} does not exist"
        except Exception as e:
            return filename, False, f"Failed to copy {filename}: {str(e)}"

    def move_val_image(self, filename: str) -> tuple[str, bool, str]:
        """
        Метод move_val_image() копирует изображение в папку images/val.
        Parameters:
            filename (str): Имя файла изображения
        Returns:
            tuple: (имя файла, успех копирования, сообщение об ошибке если есть)
        """
        source_path = os.path.join(self.path_to_images, filename)
        destination_path = os.path.join(self.destination_dir, "images", "val", filename)

        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)
                if os.path.exists(destination_path):
                    return filename, True, f"Copied {filename} to {destination_path}"
                else:
                    return filename, False, f"Failed to copy {filename}: destination file not created"
            else:
                return filename, False, f"Source file {source_path} does not exist"
        except Exception as e:
            return filename, False, f"Failed to copy {filename}: {str(e)}"

    def move_train_label(self, filename: str) -> tuple[str, bool, str]:
        """
        Метод move_train_label() копирует метку в папку labels/train.
        Parameters:
            filename (str): Имя файла метки
        Returns:
            tuple: (имя файла, успех копирования, сообщение об ошибке если есть)
     dispersed
        """
        source_path = os.path.join(self.path_to_labels, filename)
        destination_path = os.path.join(self.destination_dir, "labels", "train", filename)

        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)
                if os.path.exists(destination_path):
                    return filename, True, f"Copied {filename} to {destination_path}"
                else:
                    return filename, False, f"Failed to copy {filename}: destination file not created"
            else:
                return filename, False, f"Source file {source_path} does not exist"
        except Exception as e:
            return filename, False, f"Failed to copy {filename}: {str(e)}"

    def move_val_label(self, filename: str) -> tuple[str, bool, str]:
        """
        Метод move_val_label() копирует метку в папку labels/val.
        Parameters:
            filename (str): Имя файла метки
        Returns:
            tuple: (имя файла, успех копирования, сообщение об ошибке если есть)
        """
        source_path = os.path.join(self.path_to_labels, filename)
        destination_path = os.path.join(self.destination_dir, "labels", "val", filename)

        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, destination_path)
                if os.path.exists(destination_path):
                    return filename, True, f"Copied {filename} to {destination_path}"
                else:
                    return filename, False, f"Failed to copy {filename}: destination file not created"
            else:
                return filename, False, f"Source file {source_path} does not exist"
        except Exception as e:
            return filename, False, f"Failed to copy {filename}: {str(e)}"


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

        if len(os.listdir(self.path_to_labels)) == 0 or len(os.listdir(self.path_to_images)) == 0:
            self.logger.warning("Raw images and/or raw labels directory is empty")
            raise ValueError("Raw images and/or raw labels directory is empty")

        destination_dir_train_images = os.path.join(self.destination_dir, "images", "train")
        destination_dir_train_labels = os.path.join(self.destination_dir, "labels", "train")
        destination_dir_val_images = os.path.join(self.destination_dir, "images", "val")
        destination_dir_val_labels = os.path.join(self.destination_dir, "labels", "val")

        for dir_path in [
            destination_dir_train_images,
            destination_dir_train_labels,
            destination_dir_val_images,
            destination_dir_val_labels
        ]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    filepath = os.path.join(dir_path, file)
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        self.logger.error(f"Failed to remove {filepath}. {e}")

            else:
                os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Prepared directory {dir_path}")

        self.logger.info("Starting file filtering...")

        image_files = []
        label_files = []
        for ext in self.images_extensions:
            image_files.extend([p.name for p in Path(self.path_to_images).glob(f"*{ext}") if p.is_file()])
        for ext in self.labels_extensions:
            label_files.extend([p.name for p in Path(self.path_to_labels).glob(f"*{ext}") if p.is_file()])

        image_stems = {Path(f).stem for f in image_files}
        label_stems = {Path(f).stem for f in label_files}
        common_stems = image_stems.intersection(label_stems)

        paired_images = [f for f in image_files if Path(f).stem in common_stems]
        paired_labels = [f for f in label_files if Path(f).stem in common_stems]

        self.logger.info(f"Found {len(paired_images)} paired images and labels")

        data = {'images': paired_images, 'labels': paired_labels}
        dataset = pd.DataFrame(data=data)

        self.logger.info("Dataframe created")
        self.logger.info("Starting train-test split...")

        train_data, val_data = train_test_split(
            dataset,
            train_size=0.85,
            random_state=42,
            shuffle=True
        )

        train_data["type"] = "train"
        val_data["type"] = "val"
        dataset = pd.concat([train_data, val_data], axis=0)

        self.logger.info("Train-test split done")
        self.logger.info("Starting moving files into dataset dir...")

        copy_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for _, row in dataset.iterrows():
                filename_img = row["images"]
                filename_lbl = row["labels"]
                destination_type = row["type"]

                if destination_type == "train":
                    copy_results.append(executor.submit(self.move_train_image, filename_img))
                    copy_results.append(executor.submit(self.move_train_label, filename_lbl))
                else:
                    copy_results.append(executor.submit(self.move_val_image, filename_img))
                    copy_results.append(executor.submit(self.move_val_label, filename_lbl))

        for future in copy_results:
            filename, success, message = future.result()
            if success:
                self.logger.info(message)
            else:
                self.logger.error(message)

        train_images_count = len(os.listdir(destination_dir_train_images))
        val_images_count = len(os.listdir(destination_dir_val_images))
        train_labels_count = len(os.listdir(destination_dir_train_labels))
        val_labels_count = len(os.listdir(destination_dir_val_labels))

        expected_train = len(dataset[dataset["type"] == "train"])
        expected_val = len(dataset[dataset["type"] == "val"])

        if train_images_count == expected_train and train_labels_count == expected_train:
            self.logger.info(f"Train split verified: {train_images_count} images, {train_labels_count} labels")
        else:
            self.logger.error(
                f"Train split mismatch: {train_images_count} images, {train_labels_count} labels, expected {expected_train}")

        if val_images_count == expected_val and val_labels_count == expected_val:
            self.logger.info(f"Validation split verified: {val_images_count} images, {val_labels_count} labels")
        else:
            self.logger.error(
                f"Validation split mismatch: {val_images_count} images, {val_labels_count} labels, expected {expected_val}")

        self.logger.info("Configuration successfully completed")