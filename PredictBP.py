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

image = "img.jpg"

image = Image.open(image)
data = image.resize((180, 180))
print("size done")

model = tf.keras.models.load_model("modelScr-FLOWERS100-softmax.h5")

def inference_mage(data, class_names, model):
    img_array = tf.keras.utils.img_to_array(data)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


for i in range(30):
    inference_mage(data, class_names, model)
