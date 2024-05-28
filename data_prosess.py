import numpy as np
import cv2
import json
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def process_folder(folder_path, desired_width, desired_height):
    x_data = []
    y_data = []
    image_names = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                image_folder = os.path.join(root, "images")
                if os.path.exists(image_folder):
                    data = load_data(json_file)
                    for image_data in data['images']:
                        image_id = image_data['id']
                        image_file = os.path.join(image_folder, image_data['file_name'])
                        image_names.append(image_data['file_name'])
                        annotations = [anno for anno in data['annotations'] if anno['image_id'] == image_id]

                        image = cv2.imread(image_file)
                        resized_image = cv2.resize(image, (desired_width, desired_height))
                        resized_image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                        mask = np.zeros((desired_height, desired_width))

                        for annotation in annotations:
                            segmentation = annotation['segmentation'][0]
                            points = [(int(segmentation[i] * desired_width / image_data['width']),
                                       int(segmentation[i + 1] * desired_height / image_data['height']))
                                      for i in range(0, len(segmentation), 2)]
                            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)

                            # Малювання контурів лініями
                            cv2.polylines(mask, [np.array(points)], isClosed=True, color=0, thickness=1)

                        x_data.append(resized_image_gray)
                        y_data.append(mask.astype(np.uint8))

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    return x_data, y_data, image_names


if __name__ == "__main__":
    train_folder = './tooth_segmentation/train'
    x_train, y_train, file_names = process_folder(train_folder, 512, 512)
    print("Shape of x_train:", x_train.shape)
    print(np.unique(y_train))
    print("Shape of y_train:", y_train.shape)

    image_id = 0

    print(file_names[image_id])
    plt.subplot(1, 2, 1)
    plt.imshow(x_train[image_id])
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(y_train[image_id, :, :, 0], cmap='grey')
    plt.title('Mask')

    plt.show()

