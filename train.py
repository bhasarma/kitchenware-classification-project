#!/usr/bin/env python
# coding: utf-8

# To get started and be able to download the data:
# 
# - Join the competition and accept rules
# - Download your Kaggle credentials file
# - If you're running in Saturn Cloud, configure your instance to have access to access the kaggle credentials
# 
# 

# ## 1. Getting the Data
# 
# Execute the following cell once to download the data in Saturn Jupyter Notebook or in Saturn terminal without `!`

# In[1]:


#!kaggle competitions download -c kitchenware-classification
#!mkdir data
#!unzip kitchenware-classification.zip -d data > /dev/null
#!rm kitchenware-classification.zip


# In[2]:


get_ipython().system('ls')


# ## 2. Doiong the necessary imports

# In[3]:


import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#import cv2
from PIL import Image

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import load_img


# ## 3. Loading the dataframe

# In[4]:


df = pd.read_csv('data/train.csv', dtype={'Id': str})
df['filename'] = 'data/images/' + df['Id'] + '.jpg'
df['imagename'] = df['Id'] + '.jpg'
df.head()


# ## 4. splitting dataframe into train, test and validation

# In[5]:


train_cutoff = int(len(df) * 0.6)
full_train_cutoff = int(len(df) * 0.8)

df_train = df[:train_cutoff]
df_val = df[train_cutoff:full_train_cutoff]
df_test = df[full_train_cutoff:]
df_full_train = df[:full_train_cutoff]


# ## 5. EDA

# ### 5.1 number of images and classes

# In[6]:


len(df), len(df_train), len(df_val), len(df_test), len(df_full_train)


# In[7]:


print("for df:")
print(df.describe())

print('\n')
print("df_train:")
print(df_train.describe())

print('\n')
print("df_val:")
print(df_val.describe())

print('\n')
print("df_test:")
print(df_test.describe())

print('\n')
print("df_full_train:")
print(df_full_train.describe())


# We see that all datasets after splitting have also 6 classes

# ### 5.2 names of classes and their frequencies

# In[8]:


df.head()


# In[9]:


df['label'].nunique()


# In[10]:


df['label'].value_counts()


# In[11]:


# Get the counts of unique values in the "label" column
counts_df = df["label"].value_counts(normalize = True)
# Plot the frequencies as a histogram
counts_df.plot(kind="bar")

# Show the plot
plt.title("df")
plt.show()


# In[12]:


counts_df_train = df_train["label"].value_counts(normalize = True)
counts_df_train.plot(kind="bar")
plt.title("df_train")
plt.show()


# In[13]:


counts_df_val = df_val["label"].value_counts(normalize = True)
counts_df_val.plot(kind="bar")
plt.title("df_val")
plt.show()


# In[14]:


counts_df_test = df["label"].value_counts(normalize = True)
counts_df_test.plot(kind="bar")
plt.title("df_test")
plt.show()


# In[15]:


## plot all above plots in one plot with the below template, when you get time
# https://www.geeksforgeeks.org/bar-plot-in-matplotlib/


# ### 5.3 random visual assessment of few images

# In[16]:


df_train.head()


# In[17]:


df_train['filename'][100]


# In[18]:


#load one random image
load_img(df_train['filename'][100])


# In[19]:


path = os.getcwd()
path


# In[20]:


os.chdir(path+"/data/images/")


# In[21]:


df['imagename'] = df['Id'] + '.jpg'


# In[22]:


df.head()


# In[23]:


fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(8, 7),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    random = np.random.randint(1, len(df))
    ax.imshow(plt.imread(df.imagename[random]))
    ax.set_title(df.label[random], fontsize = 12)
plt.tight_layout(pad=0.5)
plt.show()


# ### 5.4 Distribution of images size (kb) and resolution (width and height in pixels)

# In[24]:


for image in glob.glob("*.jpg"):
    df["image_size(kB)"] = round((os.path.getsize(image)/1000),2)
    img = Image.open(image)
    width, height = img.size
    df["image_width(pixel)"]= width
    df["image_height(pixel)"]= height


# In[25]:


df.head()


# Now let's check some properties of the images.

# In[26]:


df['image_size(kB)'].describe().apply("{0:.5f}".format)


# Thus, all images are of same size in kB.

# In[27]:


df['image_width(pixel)'].describe().apply("{0:.5f}".format)


# Thus, all images are of same width i.e 750 pixels.

# In[28]:


df['image_height(pixel)'].describe().apply("{0:.5f}".format)


# Thus, all images are of same height i.e. 1000 pixels.

# In[29]:


# Let's check size of an array of an image 
img = load_img(df['imagename'][100], target_size=(299, 299))
x = np.array(img)
x.shape


# ## 6. Training with Pre-trained convolutional neural networks: Base Model 
# 
# We'll use a pre-trained CNN model (Xception) from keras applications and then we'll use transfer learning to adjust to our use case. We'll built on top of this pre-trained Imagenet model. 

# In[30]:


os.chdir(path)
os.getcwd()


# In[31]:


from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[32]:


df_train.head()


# In[33]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
)


# ### Let's create the base model first

# In[34]:


base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)
base_model.trainable = False

#let's create a new top
# inputs for base model
inputs = keras.Input(shape=(150, 150, 3))

#apply base model to inputs
base = base_model(inputs, training=False)  
vectors = keras.layers.GlobalAveragePooling2D()(base)
outputs = keras.layers.Dense(6)(vectors) #we have 6 classes

model = keras.Model(inputs, outputs)


# ### using optimizer for changing weights
# 
# In this step, model learns something for our images and tries to change the weights.

# In[35]:


# there are many other optimizers. We'll use Adam
# https://keras.io/api/optimizers/
# learning rate is similar to eta in xgboost
learning_rate = 0.01
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

#lower the loss is better it is. We have multi classificatin, so we use categricalCrossentropy
#logits=True bceause we want to keep the raw score
loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# Now it's complied and we are ready to train a model.

# In[36]:


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)


# Thus, we see that training accuracy increases with each epoch, however validation accuracy is highest for epoch = 2

# In[37]:


#let's plot accuracy on training data and on validation data
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xticks(np.arange(10))
plt.legend()


# Thus, epoch = 2 is the best echo with 87% accuracy aprx., although there is some overfitting with it. Epoch 9 is the highest but with lots of overfitting.

# ## 7. Parameter tuning by adjusting the learning rate

# In[38]:


def make_model(learning_rate=0.01):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(6)(vectors)
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


# In[39]:


scores = {}

for lr in [0.0001, 0.001, 0.01, 0.1]:
    print(lr)
    
    model = make_model(learning_rate=lr)
    history = model.fit(train_generator,epochs=10,validation_data=val_generator)
    scores[lr] = history.history

    print()
    print()


# In[40]:


for lr, hist in scores.items():
    plt.plot(hist['accuracy'], label=('train=%s' % lr))
    #plt.plot(hist['val_accuracy'], label=('val=%s' % lr))

plt.xticks(np.arange(10))
plt.legend()


# In[41]:


for lr, hist in scores.items():
    #plt.plot(hist['accuracy'], label=('train=%s' % lr))
    plt.plot(hist['val_accuracy'], label=('val=%s' % lr))

plt.xticks(np.arange(10))
plt.legend()


# In[45]:


for lr, hist in scores.items():
    plt.plot(hist['accuracy'],'--', label=('train=%s' % lr))
    plt.plot(hist['val_accuracy'],'-', label=('val=%s' % lr))

plt.xticks(np.arange(10))
plt.legend()


# In above picture, all dotted lines are for training and solid line are for validations. Thus 0.0001 is the best choice for learning rate.

# In[46]:


#best learning rate
learning_rate = 0.0001


# ## Checkpointing
# 
# Idea of checkpointing is to save only the best model (or saving a model when certain conditions are met). We'll use it for saving the best model. We'll now train the model for best learning rate, run it for 10 epochs and out of them save the best model.

# In[48]:


os.chdir(path)


# In[49]:


os.getcwd()


# In[50]:


#save the last model we trained
model.save_weights('model_v1.h5', save_format='h5')


# In[51]:


chechpoint = keras.callbacks.ModelCheckpoint(
    'xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[52]:


model = make_model(learning_rate=learning_rate)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[chechpoint]
)


# Above we see that validation accuracy improves with almost each epoch. Overfitting is very small. So as of now, we'll select the last model `xception_v1_10_0.874.h5` as the best model. We'll delete other saved model.

# ## Adding more layers
# 
# - We'll add one more ineer dense layers to our neural network 
# - We'll experiment with different sizes of inner layer

# In[53]:


def make_model(learning_rate=0.01, size_inner=100):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    #add one inner layer of size 100
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    
    outputs = keras.layers.Dense(6)(inner)
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


# In[58]:


chechpoint = keras.callbacks.ModelCheckpoint(
    'xception_v2_{epoch:02d}_{val_accuracy:.3f}.h5', #v2 means with inner layer
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[59]:


learning_rate


# In[60]:


scores = {}

for size in [10, 100, 1000]:
    print(size)

    model = make_model(learning_rate=learning_rate, size_inner=size)
    history = model.fit(train_generator, epochs=10, validation_data=val_generator,callbacks=[chechpoint])    
    scores[size] = history.history

    print()
    print()


# In[65]:


for size, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('val=%s' % size))
    plt.plot(hist['accuracy'],'--', label=('train=%s' % size))

plt.xticks(np.arange(10))
#plt.yticks([0.78, 0.80, 0.82, 0.825, 0.83])
plt.legend()


# Thus, size 10  with accuracy of 0.887 has the lowest overfitting. We'll use this value of size further. We'll delete other models saved with checkpoint. 

# In[67]:


size = 10


# We will also tune number of inner layers and size of inner layers

# ## Regularization with dropout
# 
# Here we are adding drpout to the inner layer. This is way to perform regularization.

# In[68]:


def make_model(learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
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


# In[70]:


learning_rate, size


# In[71]:


#v3 model means with dropout
chechpoint = keras.callbacks.ModelCheckpoint(
    'xception_v3_{epoch:02d}_{val_accuracy:.3f}.h5', #v2 means with inner layer
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[73]:


scores = {}

for droprate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    print(droprate)

    model = make_model(
        learning_rate=learning_rate,
        size_inner=size,
        droprate=droprate
    )

    history = model.fit(train_generator, epochs=30, validation_data=val_generator,callbacks=[chechpoint])
    scores[droprate] = history.history

    print()
    print()


# In[78]:


for droprate, hist in scores.items():
    plt.plot(hist['val_accuracy'], label=('val=%s' % droprate))

plt.ylim(0.8, 0.90)
plt.legend()


# In[79]:


# final comparison between droprate of 0.0 and 0.2
hist = scores[0.0]
plt.plot(hist['val_accuracy'], label=0.0)

hist = scores[0.2]
plt.plot(hist['val_accuracy'], label=0.2)

plt.legend()
#plt.plot(hist['accuracy'], label=('val=%s' % droprate))


# **Conclusion** here is that `droprate` doesn't improve accuracy much therefore we'll go on from here with `droprate=0.0` 

# In[82]:


droprate = 0.0


# ## Data augmentation
# 
# Here idea is to create more data from existing data and see if it improves training of images.

# In[80]:


train_generator.class_indices


# In[81]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  vertical_flip=True)

train_generator = train_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
)


# In[83]:


learning_rate, size, droprate


# In[84]:


chechpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_{epoch:02d}_{val_accuracy:.3f}.h5', #v2 means with inner layer
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[85]:


model = make_model(
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_generator, epochs=50, validation_data=val_generator,callbacks=[chechpoint])


# In[86]:


hist = history.history
plt.plot(hist['val_accuracy'], label='val')
plt.plot(hist['accuracy'], label='train')

plt.legend()


# We see above that, after about 9 or 10 epochs, both validation and training accuracy improves. But it comes at the cost of overfitting. If we consider 10 epochs, then we achieve accuracy of 0.8795. So we cankeep the model `ception_v4_09_0.879.h5`. But this accuracy is still less than `xception_v3_26_0.889.h5`. So we'll delete also v4 model.

# ## Training a larger model
# 
# Here we'll train images of size 299x99 instead of 150x150. Till now we were using 150x150 images for saving computational resources.

# In[87]:


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


# In[88]:


input_size = 299


# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#                                    shear_range=10,
#                                    zoom_range=0.1,
#                                    horizontal_flip=True)
# 
# train_generator = train_datagen.flow_from_dataframe(
#     df_train,
#     x_col='filename',
#     y_col='label',
#     target_size=(input_size, input_size),
#     batch_size=32,
# )
# 
# val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# 
# val_generator = val_datagen.flow_from_dataframe(
#     df_val,
#     x_col='filename',
#     y_col='label',
#     target_size=(input_size, input_size),
#     batch_size=32,
# )

# In[90]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_1_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[91]:


learning_rate, size, droprate


# In[92]:


model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_generator, epochs=50, validation_data=val_generator,callbacks=[checkpoint])


# In[ ]:





# From above experiment on 1. larger size of image and 2. using data augmentation with shear, zoom_range and horizontal flip, we conclude that:
# 
# - we get much better accuracy (above 0.9 ) compared to all other models.
# - validation accuracy improves almost continuously after each epoch.
# - just by visulazing, it seems that epoch=15 gives the best compromise between accuracy and overfitting with validation accuracy = 0.9541 and training accuracy = 0.9565
# - Terefore, we keep the model `xception_v4_1_15_0.954.h5` and delete other models saved by checkpoint from this experiment 

# ## Training the final model on full_train dataset
# 
# - Train the final model for full training dataset. Till now we were using df_train instead of df_full_train and df_val was not part of training. Now we'll train on df_full_train and test it on df_test
# 
# - It would be interesting to see, if more training data leads to an improve in accuracy 

# In[93]:


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

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col='filename',
    y_col='label',
    target_size=(input_size, input_size),
    batch_size=32,
)


# In[94]:


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v4_2_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)


# In[95]:


learning_rate, size, droprate


# In[97]:


input_size


# In[96]:


model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_generator, epochs=20, validation_data=val_generator,callbacks=[checkpoint])


# ### Conclusion of above section on training model with full train dataset
#     
#     - better accuracy is achieved compared to all models now and also compared to training on only df_train 
#     - overfitting starts only after epoch 19/20
#     - Best model is with epoch 16/20 and no over-fitting, test accuracy = 0.9649 and test accuracy = 0.9570
#     - we'll use this model `xception_v4_2_16_0.965.h5` for further deployment as the best model

# ### The End!
