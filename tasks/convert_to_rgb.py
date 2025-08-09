import os
import argparse
import json
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Конвертация TIFF в RGB JPEG и обработка/переименование JSON-аннотаций LabelMe с возможностью смещения номеров.")
parser.add_argument("--offset", type=int, default=0, help="Смещение для номеров файлов (по умолчанию 1000)")
args = parser.parse_args()


input_folder = "C:\Python projects\Datasets\images"
output_folder = os.path.join(input_folder, 'rgb')
labels_folder = os.path.join(input_folder, 'labels')


os.makedirs(output_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

def convert_tiff_to_rgb_jpg_and_process_json(input_dir, output_dir, labels_dir, offset):
    count_images = 0
    count_json = 0


    for filename in tqdm(os.listdir(input_dir), desc="Конвертация TIFF -> RGB JPEG"):
        input_path = os.path.join(input_dir, filename)

        if not filename.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
            continue

        try:
            with Image.open(input_path) as img:
                rgb_img = img.convert('RGB')

                base_name = os.path.splitext(filename)[0]
                try:
                    number = int(''.join(filter(str.isdigit, base_name)))
                    new_number = number + offset
                    new_base_name = f"{new_number:04d}"

                except ValueError:
                    new_base_name = f"{base_name}_offset{offset}"

                output_path = os.path.join(output_dir, f"{new_base_name}.jpg")
                rgb_img.save(output_path, format='JPEG')
                count_images += 1
        except Exception as e:
            print(f" Ошибка с {filename}: {e}")


    for filename in tqdm(os.listdir(input_dir), desc="Обработка и переименование JSON-аннотаций"):
        input_path = os.path.join(input_dir, filename)


        if not filename.lower().endswith('.json'):
            continue

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            base_name = os.path.splitext(filename)[0]
            try:
                number = int(''.join(filter(str.isdigit, base_name)))
                new_number = number + offset
                new_base_name = f"{new_number:04d}"

            except ValueError:
                new_base_name = f"{base_name}_offset{offset}"


            if 'imagePath' in data:
                old_image_name = os.path.splitext(data['imagePath'])[0]
                try:
                    old_number = int(''.join(filter(str.isdigit, old_image_name)))
                    new_image_number = old_number + offset
                    new_image_base_name = f"{new_image_number:04d}"

                except ValueError:
                    new_image_base_name = f"{old_image_name}_offset{offset}"
                new_image_path = f"{new_image_base_name}.jpg"
                data['imagePath'] = new_image_path

            elif 'imageData' in data:
                pass

            else:
                print(f"Предупреждение: Поле imagePath или imageData не найдено в {filename}")


            output_path = os.path.join(labels_dir, f"{new_base_name}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            os.remove(input_path)
            count_json += 1
        except Exception as e:
            print(f" Ошибка с {filename}: {e}")

    print(f"Сконвертировано изображений: {count_images}")
    print(f"Обработано и переименовано JSON-аннотаций: {count_json}")
    print(f"Изображения сохранены в: {output_dir}")
    print(f"JSON-аннотации сохранены в: {labels_dir}")


if __name__ == '__main__':
    convert_tiff_to_rgb_jpg_and_process_json(input_folder, output_folder, labels_folder, args.offset)