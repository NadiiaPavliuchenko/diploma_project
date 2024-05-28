import os
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import tensorflow as tf
import matplotlib.pyplot as plt

desired_width = 512
desired_height = 512
mask_generated = False


def load_models():
    global model1, classifier_model
    model1_path = './final_models/trained_attunet_final1.h5'
    classifier_model_path = './final_models/classificator.h5'

    if not os.path.exists(model1_path):
        messagebox.showerror("Error", "Segmentation model file not found.")
        return
    if not os.path.exists(classifier_model_path):
        messagebox.showerror("Error", "Classifier model file not found.")
        return

    model1 = tf.keras.models.load_model(model1_path)
    classifier_model = tf.keras.models.load_model(classifier_model_path)


def preprocess_image(image_path, desired_width, desired_height):
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (desired_width, desired_height))
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    return image_gray.reshape((desired_height, desired_width, 1))


def predict_with_timing(model, image):
    start_time = time.time()
    prediction = model.predict(image)
    print(prediction.max())
    if prediction.max() < 0.7:
        print("Low model confidence. Segmentation failed.")
        return None

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Prediction time: {elapsed_time} s")
    return prediction, elapsed_time


def predict_image(image):
    global mask_generated
    processed_image = preprocess_image(image, desired_width, desired_height)

    # Класифікація зображення
    classification_result = classifier_model.predict(np.array([processed_image]))
    if classification_result[0][0] >= 0.5:
        messagebox.showinfo("Result", "The picture is not a picture of the jaw.")
        return

    # Сегментація зображення
    prediction1, time1 = predict_with_timing(model1, np.array([processed_image]))

    if prediction1 is not None:
        save_mask(prediction1[0, :, :, 0], os.path.join('./masks', 'mask1.png'))
        mask_generated = True


def save_mask(mask, file_path):
    plt.imsave(file_path, mask)


def predict_and_save_masks():
    global image_path
    if not image_path:
        messagebox.showerror("Error", "Please load an image first.")
        return
    predict_image(image_path)


def open_file(event=None):
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        load_image(image_path)
    else:
        messagebox.showerror("Error", "Please select an image file.")


def show_masks():
    global mask_generated
    if not mask_generated:
        messagebox.showerror("Error", "Please predict masks first.")
        return
    mask1_path = './masks/mask1.png'
    if not os.path.exists(mask1_path):
        messagebox.showerror("Error", "Mask file not found.")
        return
    mask1 = Image.open(mask1_path)
    mask1.show()


def drop_image(event):
    global image_path
    file_path = event.data
    file_path = file_path.strip('{}')
    load_image(file_path)


def load_image(file_path):
    global image_path
    try:
        image_path = file_path
        image = Image.open(file_path)
        image_resized = image.resize((desired_width, desired_height))
        image_gray = image_resized.convert('L')
        photo = ImageTk.PhotoImage(image_gray)
        dbox.config(image=photo)
        dbox.image = photo
    except Exception as e:
        messagebox.showerror("Error", f"Error loading image: {e}")


def show_help_window():
    help_text = """
    To use the application:
        1. Upload an image using the "Load Image" button or drag it into the field.
        2. Click the "Predict & Save Mask" button to obtain masks using model.
        3. The "Show Mask" button allows you to view the result mask.
    """
    messagebox.showinfo("Help", help_text)


root = TkinterDnD.Tk()
root.title("Mask Predictor")
root.geometry("600x600")

menubar = tk.Menu(root)
menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Menu', menu=menu)
menu.add_command(label='Help', command=show_help_window)

dbox = tk.Label(root, text="Drop your x-ray here", bg="white", bd=2, relief="groove", width=512, height=512)
dbox.pack(expand=True, fill='both')

# Register drag-and-drop targets and handle drop event
dbox.drop_target_register(DND_FILES)
dbox.dnd_bind('<<Drop>>', drop_image)

load_button = tk.Button(dbox, text="Load Image", command=open_file)
load_button.place(relx=0.5, rely=0.58, anchor="center")
load_button.lower()

load_models()

if not os.path.exists('./masks'):
    os.makedirs('./masks')

# Creating a frame to hold buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, pady=10)

predict_button = tk.Button(button_frame, text="Predict & Save Mask", command=predict_and_save_masks)
predict_button.pack(side=tk.LEFT, padx=5)

show_button = tk.Button(button_frame, text="Show Mask", command=show_masks)
show_button.pack(side=tk.LEFT, padx=5)

root.config(menu=menubar)
root.mainloop()
