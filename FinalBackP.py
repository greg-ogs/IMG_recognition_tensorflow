# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:06:12 2021

@author: grego
Gregorio ALejandro Oropeza Gomez
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

#Image size, target
image_size = (180, 180)
batch_size = 32

print('Introdusca el directorio donde se encuentra el dataset, recuerde usar "/"')
data_dir = input() #For classes

# IMages for validation & training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
class_names = train_ds.class_names
print(class_names)

# Argumentation 
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)

#Performance
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

input_shape=image_size + (3,)
print("Ingrese el numero de clases que se mostraron en pantalla")
num_classes=input()

#define model input_shape & num_classes
inputs = keras.Input(shape=input_shape)
    # Argumentation (Again)
x = data_augmentation(inputs)

x = layers.Rescaling(1.0 / 255)(x)#reescalado
x = layers.Conv2D(32, 3, strides=2, padding="same")(x)#Layers
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

x = layers.Conv2D(64, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

previous_block_activation = x  

for size in [128, 256, 512, 728]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

x = layers.SeparableConv2D(1024, 3, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.GlobalAveragePooling2D()(x)

if num_classes == 2:
    activation = "sigmoid"
    units = 1
else:
    activation = "softmax"
    units = num_classes

x = layers.Dropout(0.5)(x)
outputs = layers.Dense(units, activation=activation)(x)
    #return keras.Model(inputs, outputs)#Se puede reconstruir las capas como una funcion

#Funcion model(in & out)
model = keras.Model(inputs,outputs)
keras.utils.plot_model(model, show_shapes=True)
print('Introdusca la cantidad de epocas')
epochs = input()
epochs = int(epochs)
#CallB
callbacks = [
    keras.callbacks.ModelCheckpoint("epoch_{epoch}_save.h5"),
]

#Compile
model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False, name ="SGD"),
    #se pueden elegir diferentes funciones de perdida y optimizadores en esta parte
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.AUTO,name='sparse_categorical_crossentropy'),#SUM20%aprox---AUTO80%aprox 
    metrics=["accuracy"],
)

history=model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
#Save model
model.save('modelScr100.h5')
#print model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precicion')
plt.plot(epochs_range, val_acc, label='Validacion de precision')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perdida')
plt.plot(epochs_range, val_loss, label='Validacion de la perdida')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()