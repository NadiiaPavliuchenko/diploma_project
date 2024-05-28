from data_prosess import process_folder
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import time
import tensorflow.keras as keras
from keras_unet_collection import base
from model import custom_unet
from keras.models import *
from keras.layers import *
from tensorflow.keras.utils import Sequence
import numpy as np


n_classes = 1

# model = custom_unet()
def get_base_unet_model(input_shape):
    inputs = Input(input_shape)
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


def modified_unet(input_size=(512, 512, 1), num_classes=1):
    inputs = Input(input_size)

    base_output = BaseUnetBlock(input_size)(inputs)

    X = Conv2D(64, (3, 3), padding='same')(base_output)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)


    final_conv = Conv2D(num_classes, (1, 1), activation='sigmoid')(X)

    model = Model(inputs, final_conv)

    return model



model = modified_unet()


model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

model.summary()

train_folder = './tooth_segmentation/train'
x_train, y_train, file_names = process_folder(train_folder, 512, 512)
valid_folder = './tooth_segmentation/valid'
x_valid, y_valid, val_file_names = process_folder(valid_folder, 512, 512)


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


train_gen = DataGenerator(x_train, y_train, 16)
test_gen = DataGenerator(x_valid, y_valid, 16)

results = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=16, epochs=1, verbose=1)

model.save_weights('./content/trained_unet.weights.h5')

t = time.time()
my_keras_model_filepath = './content/trained_unet.h5'.format(int(t))
model.save(my_keras_model_filepath)


training_loss = results.history['loss']
training_accuracy = results.history['accuracy']
training_dice_coef = results.history['dice_coef']
validation_loss = results.history['val_loss']
validation_accuracy = results.history['val_accuracy']
validation_dice_coef = results.history['val_dice_coef']

epochs_range = range(len(training_accuracy))
fig, axes = plt.subplots(1, 4, figsize=(16, 4))


axes[0].plot(epochs_range, training_accuracy, label='Training Accuracy')
axes[0].plot(epochs_range, validation_accuracy, label='Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()


axes[1].plot(epochs_range, training_loss, label='Training Loss')
axes[1].plot(epochs_range, validation_loss, label='Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

axes[2].plot(epochs_range, training_dice_coef, label='Training Dice')
axes[2].plot(epochs_range, validation_dice_coef, label='Validation Dice')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Dice Coefficient')
axes[2].legend()

# Відобразити всі графіки разом
plt.tight_layout()
plt.show()
