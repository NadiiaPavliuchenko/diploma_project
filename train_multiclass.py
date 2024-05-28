import os
import keras
from keras  import metrics
import cv2
import time
import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from keras_unet_collection import base, losses
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from model import custom_unet
from tensorflow.keras.utils import Sequence


n_classes=33
classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# def plot_masks_with_image(image, masks):
#     num_classes = np.max(masks) + 1
#     class_names = [f"Клас {i}" for i in range(num_classes)]
#
#     fig, axs = plt.subplots(1, num_classes + 1, figsize=(15, 5))
#
#
#     axs[0].imshow(image, cmap='gray')
#     axs[0].set_title('Оригінальне зображення')
#
#
#     for i in range(num_classes):
#         masked_image = image.copy()
#         class_mask = (masks == i).astype(np.uint8) * 255
#         axs[i+1].imshow(class_mask, cmap='jet')
#         axs[i+1].set_title(class_names[i])
#
#     plt.show()

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'skyblue', 'lime', 'indigo', 'peru', 'tan', 'gold', 'coral', 'orchid', 'teal', 'navy', 'salmon', 'khaki', 'violet', 'crimson', 'turquoise', 'sienna', 'plum', 'aquamarine']

def plot_masks_with_image(image, masks):
    num_classes = np.max(masks) + 1
    class_names = [f"Клас {i}" for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.imshow(image, cmap='gray')
    ax.set_title('Оригінальне зображення')

    background_color = 'white'

    cmap_colors = [colors[i] for i in range(num_classes)]

    for i in range(1, num_classes):
        class_mask = (masks == i).astype(np.uint8)
        ax.imshow(np.ma.masked_where(class_mask == 0, class_mask), cmap=ListedColormap(cmap_colors[i]), alpha=0.5, vmin=0, vmax=num_classes-1)

    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(1, num_classes)]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    ax.set_facecolor(background_color)

    plt.show()


def create_instance_mask(image_size, annotations):
    mask = np.zeros(image_size[:2], dtype=np.uint8)
    for annotation in annotations:
        class_id = annotation['category_id']
        segmentation = annotation['segmentation'][0]
        points = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [points], color=class_id)

    return mask[..., np.newaxis]

def process_folder(folder_path, annotations):
    x_data = []
    y_data = []
    image_names = []

    images = annotations.get("images", [])

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                image_names.append(file)

                image_annotations = None
                for image in images:
                    if image.get("file_name") == file:
                        image_id = image.get("id")
                        image_annotations = [anno for anno in annotations.get("annotations", [])
                                             if anno.get("image_id") == image_id]
                        break

                if image_annotations is None:
                    continue

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (512, 512))


                instance_mask = create_instance_mask((640, 640), image_annotations)
                instance_mask = cv2.resize(instance_mask, (512, 512), interpolation=cv2.INTER_NEAREST)


                x_data.append(image[..., np.newaxis])
                y_data.append(instance_mask)

    return x_data, y_data, image_names

if __name__ == '__main__':
    valid_annotations = load_data("./tooth_segmentation/valid/_annotations.coco.json")
    annotations = load_data("./tooth_segmentation/test/_annotations.coco.json")

    folder_path = "./tooth_segmentation/test/images/"
    x_data, y_data, file_names = process_folder(folder_path, annotations)
    valid_folder = './tooth_segmentation/valid/images/'
    x_valid, y_valid, valid_file_names = process_folder(valid_folder, valid_annotations)

    x_data = np.array(x_data)
    y_data_expanded = [np.expand_dims(mask, axis=-1) for mask in y_data]
    y_data = np.array(y_data)

    x_valid = np.array(x_valid)
    y_valid_expanded = [np.expand_dims(mask, axis=-1) for mask in y_valid]
    y_valid = np.array(y_valid_expanded)


    image_index = 0
    print(file_names[image_index])
    plot_masks_with_image(x_data[image_index], y_data[image_index])

    train_masks_cat = to_categorical(y_data, num_classes=n_classes)
    y_train_cat = train_masks_cat.reshape((y_data.shape[0], y_data.shape[1], y_data.shape[2], n_classes))
    valid_masks_cat = to_categorical(y_valid, num_classes=n_classes)
    y_valid_cat = valid_masks_cat.reshape((y_valid.shape[0], y_valid.shape[1], y_valid.shape[2], n_classes))

    # model = custom_unet()
    def get_base_unet_model(input_shape):
        inputs = Input(input_shape)

        # Attention U-net
        outputs = base.att_unet_2d_base(inputs,
                                        filter_num=[16, 32, 64, 128, 256],
                                        stack_num_down=2,
                                        stack_num_up=2,
                                        attention='add',
                                        atten_activation='ReLU',
                                        activation='ReLU',
                                        batch_norm=True,
                                        pool=False,
                                        unpool='bilinear')

        # U-net++
        # outputs = base.unet_plus_2d_base(inputs,
        #                                  filter_num=[16, 32, 64, 128, 256],
        #                                  stack_num_down=3,
        #                                  stack_num_up=2,
        #                                  activation='ReLU',
        #                                  batch_norm=True,
        #                                  pool=True,
        #                                  unpool=True)

        return Model(inputs, outputs)


    class BaseUnetBlock(Layer):
        def __init__(self, input_shape, **kwargs):
            super(BaseUnetBlock, self).__init__(**kwargs)
            self.base_unet = get_base_unet_model(input_shape)

        def call(self, inputs):
            return self.base_unet(inputs)

        def compute_output_shape(self, input_shape):
            return self.base_unet.compute_output_shape(input_shape)


    def modified_unet(input_size=(512, 512, 1), num_classes=n_classes):
        inputs = Input(input_size)

        base_output = BaseUnetBlock(input_size)(inputs)

        X = Conv2D(64, (3, 3), padding='same')(base_output)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        final_conv = Conv2D(num_classes, (1, 1), activation='sigmoid')(X)

        model = Model(inputs, final_conv)

        return model


    model = modified_unet()

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy', losses.dice_coef]
                  )

    model.summary()

    class DataGenerator(Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y


    train_gen = DataGenerator(x_data, y_train_cat, 16)
    test_gen = DataGenerator(x_valid, y_valid_cat, 16)

    results = model.fit(train_gen, validation_data=test_gen, epochs=30, verbose=1)

    model.save_weights('./content/test.weights.h5')

    t = time.time()
    my_keras_model_filepath = './content/test.h5'.format(int(t))
    model.save(my_keras_model_filepath)

    training_loss = results.history['loss']
    training_accuracy = results.history['accuracy']
    training_dice_coef = results.history['dice_coef']
    validation_loss = results.history['val_loss']
    validation_accuracy = results.history['val_accuracy']
    validation_dice_coef = results.history['val_dice_coef']

    epochs_range = range(len(training_accuracy))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs_range, training_accuracy, label='Training Accuracy')
    axes[0, 0].plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()

    axes[0, 1].plot(epochs_range, training_loss, label='Training Loss')
    axes[0, 1].plot(epochs_range, validation_loss, label='Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()

    axes[1, 0].plot(epochs_range, training_dice_coef, label='Training Dice')
    axes[1, 0].plot(epochs_range, validation_dice_coef, label='Validation Dice')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Coefficient')
    axes[1, 0].legend()

    plt.tight_layout()
    plt.show()
