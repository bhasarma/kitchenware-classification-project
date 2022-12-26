#!/usr/bin/env python
# coding: utf-8

# It is assumed that you have already downloaded dataset from kaggle


# importing the necessary packages
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img



from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#parameters
learning_rate = 0.0001
size = 10
droprate = 0.0
input_size = 299

# loading the dataframe


df = pd.read_csv('data/train.csv', dtype={'Id': str})
df['filename'] = 'data/images/' + df['Id'] + '.jpg'
df['imagename'] = df['Id'] + '.jpg'


# splitting dataframe into train, test and validation
train_cutoff = int(len(df) * 0.6)
full_train_cutoff = int(len(df) * 0.8)

df_train = df[:train_cutoff]
df_val = df[train_cutoff:full_train_cutoff]
df_test = df[full_train_cutoff:]
df_full_train = df[:full_train_cutoff]


def make_model(input_size=150, learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(6)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model



# train on full_train
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=10,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_dataframe(
    df_full_train,
    x_col='filename',
    y_col='label',
    target_size=(input_size, input_size),
    batch_size=32,
)


# test on test dataset
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col='filename',
    y_col='label',
    target_size=(input_size, input_size),
    batch_size=32,
)

#use checkpoit to save the best performing model
checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_2_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# create model and train it
model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_generator, epochs=20, validation_data=val_generator,callbacks=[checkpoint])
##end