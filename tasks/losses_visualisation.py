import os
import pandas as pd
import matplotlib.pyplot as plt

init_path = os.path.abspath(__file__)

def get_project_root(start_path):
    current = start_path
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Project root was not found")

project_root_path = get_project_root(init_path)

results_filepath = os.path.join(project_root_path, "runs", "train", "exp", "results.csv")

# Загрузка данных из CSV
df = pd.read_csv(results_filepath)

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Box Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/seg_loss'], label='Train Seg Loss')
plt.plot(df['epoch'], df['val/seg_loss'], label='Val Seg Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Segmentation Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/cls_loss'] + df['train/box_loss'] + df['train/seg_loss'], label='Train Total Loss')
plt.plot(df['epoch'], df['val/cls_loss'] + df['val/box_loss'] + df['val/seg_loss'], label='Val Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Total Loss (Cls + Box + Seg)')
plt.legend()
plt.grid()
plt.show()