import tensorflow as tf
import numpy as np
from train_multiclass import process_folder, load_data, plot_masks_with_image
from keras_unet_collection import losses
import tensorflow.keras as keras
import time
import matplotlib.image as mpimg
import cv2

num_classes = 33


def get_model_time(model):
    start_time = time.time()
    prediction = model.predict(np.expand_dims(x_test[0], axis=0))
    end_time = time.time()

    execution_time = end_time - start_time

    print("Час виконання моделі: {:.2f} секунд".format(execution_time))


def get_metrics(model, x_test, y_test_onehot):
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy', losses.dice_coef])

    loss, accuracy, dice_coef = model.evaluate(x_test, y_test_onehot)

    print("Точність на тестових даних: {:.2f}%".format(accuracy * 100))
    print("Значення функції втрат: {:.4f}".format(loss))
    print("Значення коефіцієнту подібності Дайса: {:.2f}%".format(dice_coef * 100))

    predictions = model.predict(x_test)

    # Get IoU on test data
    m = tf.keras.metrics.OneHotMeanIoU(num_classes=num_classes)
    m.update_state(y_true=y_test_onehot, y_pred=predictions)
    print("Значення MeanIoU:", m.result().numpy())


def predict_image(model, image):
    predicted_mask = model.predict(image[np.newaxis, ..., np.newaxis])
    predicted_mask = np.argmax(predicted_mask, axis=-1)
    return predicted_mask


def get_binary_mask(model):
    image_path = './tooth_segmentation/test/images/76_jpg.rf.2199cbda75a4f2effd77818e486dbb59.jpg'
    image = mpimg.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized_gray_image = cv2.resize(gray_image, (512, 512))
    resized_gray_image = resized_gray_image[..., np.newaxis]
    predicted_mask = predict_image(model, resized_gray_image)

    normalized_mask = (predicted_mask * 255).astype(np.uint8)

    _, binary_mask = cv2.threshold(normalized_mask, 0, 255, cv2.THRESH_BINARY)
    normalized_mask = np.squeeze(normalized_mask, axis=0)
    plot_masks_with_image(resized_gray_image, normalized_mask)


if __name__ == '__main__':
    model = tf.keras.models.load_model('./final_models/trained_unet_enum_6.h5')

    annotations = load_data("./tooth_segmentation/test/_annotations.coco.json")
    test_folder = './tooth_segmentation/test/images/'
    x_test, y_test, file_names = process_folder(test_folder, annotations)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    y_test_onehot = tf.one_hot(tf.cast(y_test, dtype=tf.uint8), depth=num_classes)

    get_metrics(model, x_test, y_test_onehot)
    get_model_time(model)
    # one image
    get_binary_mask(model)
