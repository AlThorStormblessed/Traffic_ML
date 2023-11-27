from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras

from keras import optimizers, Model, layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.layers import Input, Lambda, Dense, Flatten, Dropout,BatchNormalization, GlobalAveragePooling2D, Conv2D, MaxPool2D, Activation
from keras.models import Sequential
from keras.applications import MobileNet, ResNet50, xception
from keras.utils import to_categorical
from livelossplot import PlotLossesKeras

img_height = 250
img_width = 250
batch_size = 64

training_ds = keras.preprocessing.image_dataset_from_directory(
    'data/train/',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size
)

testing_ds = keras.preprocessing.image_dataset_from_directory(
    'data/test/',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size
)

validation_ds = keras.preprocessing.image_dataset_from_directory(
    'data/val/',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size
)

print(training_ds)

AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)

def get_model():
    model=Sequential(
        [ layers.BatchNormalization(),
          layers.Conv2D(32, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(128, 3, activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(256, activation='relu'),
          layers.Dense(2, activation= 'softmax')]
      )

    # base_model = MobileNet(weights = "imagenet", include_top = False, input_shape = (48, 48, 3) )

    #base_model.trainable = False

    # layers.base_model)



    return model

model = get_model()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_ds, validation_data = validation_ds, epochs = 10)

model.save("traffic/Saved_model/model")

AccuracyVector = []
class_names = ["Accident", "Not Accident"]
plt.figure(figsize=(30, 30))
for images, labels in testing_ds.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []

    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))

    AccuracyVector = np.array(prdlbl) == labels
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Pred: ' + predlabel[i] + ' actl:' + class_names[labels[i]] )
        plt.axis('off')
        plt.grid(True)

frames = validation_ds = keras.preprocessing.image_dataset_from_directory(
    'Frame/',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size
)

for images, labels in frames.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []

    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))

    # for i in range(40):
    #     ax = plt.subplot(4, 10, i + 1)
    #     plt.imshow(images[i].numpy().astype("uint8"))
    #     plt.title('Pred: ' + predlabel[i])
    #     plt.axis('off')
    #     plt.grid(True)