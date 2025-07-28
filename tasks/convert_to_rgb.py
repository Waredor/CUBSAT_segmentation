import os
from PIL import Image
from tqdm import tqdm

# Папки
input_folder = "D:/Python projects/CUBSAT_Dataset_segmentation/images/"
output_folder = os.path.join(input_folder, 'rgb')

# Создаем папку, если её нет
os.makedirs(output_folder, exist_ok=True)


def convert_tiff_to_rgb_jpg(input_dir, output_dir):
    count = 0
    for filename in tqdm(os.listdir(input_dir), desc="Конвертация TIFF -> RGB JPEG"):
        input_path = os.path.join(input_dir, filename)

        # Пропускаем не-TIFF файлы
        if not filename.lower().endswith(('.tif', '.tiff')):
            continue

        try:
            with Image.open(input_path) as img:
                rgb_img = img.convert('RGB')

                # Заменяем расширение на .jpg
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}.jpg")

                rgb_img.save(output_path, format='JPEG')
                count += 1
        except Exception as e:
            print(f" Ошибка с {filename}: {e}")

    print(f"Сконвертировано изображений: {count}")
    print(f"Сохранены в: {output_dir}")


if __name__ == '__main__':
    convert_tiff_to_rgb_jpg(input_folder, output_folder)
