import tensorflow as tf
import numpy as np
from data_prosess import process_folder
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import time
import tensorflow.keras as keras
from keras_unet_collection import losses


# Model evaluation on test data
def get_model_time(model, x_test):
    start_time = time.time()
    prediction = model.predict(np.expand_dims(x_test[0], axis=0))
    end_time = time.time()

    execution_time = end_time - start_time

    print("Час виконання моделі: {:.2f} секунд".format(execution_time))


def get_test_metrics(model, y_test):
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy', losses.dice_coef])

    y_test = y_test.astype(np.float32)

    loss, accuracy, dice_coef = model.evaluate(x_test, y_test)

    print("Точність на тестових даних: {:.2f}%".format(accuracy * 100))
    print("Значення функції втрат: {:.4f}".format(loss))
    print("Значення коефіцієнту подібності Дайса: {:.2f}%".format(dice_coef * 100))


def plot_prediction_result(model, x_test):
    predictions = []
    for i in range(3, 6):
        prediction = model.predict(np.expand_dims(x_test[i], axis=0))
        predictions.append(prediction)

    plt.figure(figsize=(15, 15))
    for i in range(3):
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(x_test[i+3])
        plt.title('Input Image')

        plt.subplot(3, 3, i*3 + 2)
        plt.imshow(y_test[i+3][:, :, 0], cmap='gray')  # True mask
        plt.title('Real Mask')

        plt.subplot(3, 3, i*3 + 3)
        plt.imshow(predictions[i][0, :, :, 0], cmap='gray')  # Prediction for image
        plt.title('Predicted Mask')

    plt.tight_layout()
    plt.show()


def predict_one_image(model, x_test):
    image_index = 0
    prediction = model.predict(np.expand_dims(x_test[image_index], axis=0))

    plt.subplot(1, 2, 1)
    plt.imshow(x_test[image_index])
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(prediction[0, :, :, 0], cmap='gray')
    plt.title('Predicted Mask')

    plt.show()


if __name__ == '__main__':
    model = tf.keras.models.load_model('./final_models/trained_attunet_final1.h5')
    test_folder = './tooth_segmentation/test'
    x_test, y_test, file_names = process_folder(test_folder, 512, 512)

    get_model_time(model, x_test)
    get_test_metrics(model, y_test)
    # plot_prediction_result(model, x_test)
    predict_one_image(model, x_test)


