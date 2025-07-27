import os
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

PATH_TO_LABELS = 'D:\\Python projects\\CUBSAT_dataset_segmentation\\train_test_split\\labels\\'
PATH_TO_IMAGES = 'D:\\Python projects\\CUBSAT_dataset_segmentation\\train_test_split\\images\\'

target_extensions = ['.txt']
images_extensions = ['.jpg']
target_filenames = []
images_filenames = []

for el in os.listdir(PATH_TO_LABELS):
    print(el)
    if os.path.isfile(PATH_TO_LABELS + el):
        if Path(el).suffix in target_extensions:
            target_filenames.append(el)

for el in os.listdir(PATH_TO_IMAGES):
    if os.path.isfile(PATH_TO_IMAGES + el):
        if Path(el).suffix in images_extensions:
            images_filenames.append(el)

data_X = {'images': images_filenames}
data_y = {'labels': target_filenames}
dataset_X = pd.DataFrame(data=data_X)
dataset_y = pd.DataFrame(data=data_y)

X_train, y_train, X_test, y_test = train_test_split(
dataset_X,
    dataset_y,
    test_size = 0.15,
    random_state=42,
    shuffle=True
)

print(X_train)
