# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:06:12 2021

@author: grego
Gregorio ALejandro Oropeza Gomez
"""
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


#image size, target 
image_size = (180, 180)
batch_size = 32

print('Introdusca el directorio donde se encuentra el dataset, recuerde usar "/"')
data_dir = input() #For classes

#En este caso el unico objetivo del siguiente bloque es leer las clases
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(180, 180),
  batch_size=32)

class_names = train_ds.class_names 
# Aqui termina el bloque anteriormente menvionado
print('Introdusca la direccion del modelo a usar, recuerde usar "/"')#ssoftmax
modelToUse = input()

model = load_model(modelToUse) #load model

print('Introdusca la direccion donde se encuentra la imagen a identificar, recuerde usar "/"')
image = input()
img = keras.preprocessing.image.load_img(
    image, target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print('--------------------------------------------------------')
print(
    "La imagen pertenece a la calse {} con un {:.2f} % de precision"
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
print('--------------------------------------------------------')