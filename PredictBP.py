# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:06:12 2021

@author: grego
Gregorio ALejandro Oropeza Gomez
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

image_size = (180, 180)
batch_size = 32

data_dir = "E:/IMG_recognition_tensorflow/flower_photos/flower_photos"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32)

class_names = train_ds.class_names
print(class_names)

image = "img.jpg"

image = Image.open(image)

model = tf.keras.models.load_model("modelScr-FLOWERS100-softmax.h5")


def inference_mage(image, class_names, model_to_inference):
    data = image.resize((180, 180))
    print("size done")
    img_array = tf.keras.utils.img_to_array(data)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model_to_inference.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def convert(model_to_convert):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)  # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

def read_tflite(model_tfl, image):
    interpreter = tf.lite.Interpreter(model_tfl)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    print(input_shape)
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = image.resize((input_shape[1], input_shape[2]))
    print("size done")
    img_array = tf.keras.utils.img_to_array(input_data)
    img_array = tf.expand_dims(img_array, 0)
    interpreter.set_tensor(input_details[0]['index'], img_array)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    score = tf.nn.softmax(output_data[0])
    print(score)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

for i in range(1):
    # inference_mage(image, class_names, model)
    # convert(model)
    read_tflite("model.tflite", image)
