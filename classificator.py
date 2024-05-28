import os
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    categories = {category['id']: category['name'] for category in data['categories']}
    images = {image['id']: image for image in data['images']}
    annotations = data['annotations']
    return categories, images, annotations


def get_class_ids(categories, class_names):
    return [category_id for category_id, category_name in categories.items() if category_name in class_names]


def load_and_preprocess_image(image_path, target_size=(512, 512)):
    image = Image.open(image_path).convert('L')  # Перетворення в чорно-біле
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Нормалізація
    return image_array


def create_dataset(images, annotations, class_ids, image_dir):
    image_paths = []
    labels = []

    for annotation in annotations:
        if annotation['category_id'] in class_ids:
            image_id = annotation['image_id']
            image_info = images[image_id]
            image_path = os.path.join(image_dir, image_info['file_name'])
            image_paths.append(image_path)
            labels.append(class_ids.index(annotation['category_id']))

    return image_paths, labels


# Шляхи до анотацій та зображень для кожної вибірки
train_annotation_file = './x-ray_binary/train/_annotations.coco.json'
val_annotation_file = './x-ray_binary/valid/_annotations.coco.json'
test_annotation_file = './x-ray_binary/test/_annotations.coco.json'

train_image_dir = './x-ray_binary/train/images/'
val_image_dir = './x-ray_binary/valid/images/'
test_image_dir = './x-ray_binary/test/images/'

# Зчитування анотацій
train_categories, train_images, train_annotations = load_coco_annotations(train_annotation_file)
val_categories, val_images, val_annotations = load_coco_annotations(val_annotation_file)
test_categories, test_images, test_annotations = load_coco_annotations(test_annotation_file)

# Імена класів, які вас цікавлять
class_names = ['jaws_xrays', 'non_jaw_xrays']

# Отримання ID класів
class_ids = get_class_ids(train_categories, class_names)

# Підготовка даних для тренувальної вибірки
train_image_paths, train_labels = create_dataset(train_images, train_annotations, class_ids, train_image_dir)
train_images = np.array([load_and_preprocess_image(path) for path in train_image_paths])
train_labels = np.array(train_labels)

# Підготовка даних для валідаційної вибірки
val_image_paths, val_labels = create_dataset(val_images, val_annotations, class_ids, val_image_dir)
val_images = np.array([load_and_preprocess_image(path) for path in val_image_paths])
val_labels = np.array(val_labels)

# Підготовка даних для тестової вибірки
test_image_paths, test_labels = create_dataset(test_images, test_annotations, class_ids, test_image_dir)
test_images = np.array([load_and_preprocess_image(path) for path in test_image_paths])
test_labels = np.array(test_labels)


def create_tf_dataset(images, labels, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


train_dataset = create_tf_dataset(train_images, train_labels)
val_dataset = create_tf_dataset(val_images, val_labels)
test_dataset = create_tf_dataset(test_images, test_labels)


def build_classification_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    X = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    X = tf.keras.layers.MaxPooling2D((2, 2))(X)
    X = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(X)
    X = tf.keras.layers.MaxPooling2D((2, 2))(X)
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(X)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


classification_model = build_classification_model((512, 512, 1))
# classification_model.fit(train_dataset, epochs=10, validation_data=val_dataset)
#
# # Оцінка моделі


model = tf.keras.models.load_model('./classificator (1).h5')

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# loss, accuracy = model.evaluate(test_dataset)
# print(f'Test accuracy: {accuracy}')
predictions = model.predict(test_images)


num_images = 5
random_indices = np.random.choice(len(test_images), num_images, replace=False)


fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

for i, idx in enumerate(random_indices):
    img = test_images[idx].reshape(512, 512)
    true_label = 'Jaws X-rays' if test_labels[idx] == 0 else 'Non-Jaw X-rays'
    predicted_label = 'Jaws X-rays' if predictions[idx] < 0.5 else 'Non-Jaw X-rays'

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"True: {true_label}\nPredicted: {predicted_label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
