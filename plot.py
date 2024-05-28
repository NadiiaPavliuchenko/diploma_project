import pandas as pd
import cv2
import json
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# побудова порівняльного графіка точності, функції втрат та коефіцієнту Дайса трьох моделей
# за результатами їх навчання по епохам

# df = pd.read_excel('./results_with_classes4.xlsx')
#
# grouped_data = df.groupby('Model')
#
# # Побудова графіку для кожної моделі
# for model, data in grouped_data:
#     plt.plot(data['Epoch'], data['Val-accuracy'], label=f'{model}')
#
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy Comparison')
# plt.legend()
# plt.show()
#
#
# for model, data in grouped_data:
#     plt.plot(data['Epoch'], data['Val-loss'], label=f'{model}')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Comparison')
# plt.legend()
# plt.show()

# for model, data in grouped_data:
#     plt.plot(data['Epoch'], data['Val-dice_coef'], label=f'{model}')
#
# plt.xlabel('Epoch')
# plt.ylabel('Dice coefficient')
# plt.title('Dice Comparison')
# plt.legend()
# plt.show()

train_path = './x-ray_binary/train'
test_path = './x-ray_binary/test'
valid_path = './x-ray_binary/valid'


# побудова гістограми роподілу розмірів зображень в датасеті
def get_image_sizes(dataset_path):
    image_sizes = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                height, width, _ = image.shape
                image_sizes.append((width, height))
    return image_sizes


def plot_image_sizes():
    train_image_sizes = get_image_sizes(train_path)
    test_image_sizes = get_image_sizes(test_path)
    valid_image_sizes = get_image_sizes(valid_path)

    all_image_sizes = train_image_sizes + test_image_sizes + valid_image_sizes

    plt.hist(all_image_sizes, bins=30, edgecolor='black')
    plt.xlabel('Ширина зображення')
    plt.ylabel('Кількість зображень')
    plt.title('Розподіл розмірів зображень у датасеті')
    plt.show()


# побудова гістограми розподілу зображень в датасеті за класами
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def count_images_per_class(dataset_path):
    class_counts = {}
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(root, file)
                data = load_data(json_file)
                for annotation in data['annotations']:
                    class_id = annotation['category_id']
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                    else:
                        class_counts[class_id] = 1
    return class_counts


def plot_image_classes():
    train_class_counts = count_images_per_class(train_path)
    test_class_counts = count_images_per_class(test_path)
    valid_class_counts = count_images_per_class(valid_path)

    all_class_counts = {}

    for class_id in set().union(train_class_counts, test_class_counts, valid_class_counts):
        all_class_counts[class_id] = train_class_counts.get(class_id, 0) + \
                                     test_class_counts.get(class_id, 0) + \
                                     valid_class_counts.get(class_id, 0)

    plt.bar(all_class_counts.keys(), all_class_counts.values(), color='skyblue')
    plt.xlabel('Класи')
    plt.ylabel('Кількість зображень')
    plt.title('Розподіл кількості зображень до класів у датасеті')
    plt.show()


def plot_all_metrics():
    df = pd.read_excel('./results_with_classes4.xlsx')

    grouped_data = df.groupby('Model')

    models = grouped_data.groups.keys()

    # Побудова графіків для кожної метрики
    metrics = ['Val-accuracy', 'Val-loss', 'Val-dice_coef']
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    for i, metric in enumerate(metrics):
        for model_name, model_data in grouped_data:
            data = model_data
            axes[i].plot(data['Epoch'], data[metric], label=f'{model_name}')

        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric)
        axes[i].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # plot_image_sizes()
    plot_image_classes()
    # plot_all_metrics()
