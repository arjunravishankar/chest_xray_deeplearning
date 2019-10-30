import pandas as pd
import numpy as np
import os
# from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# path = './data/train_p10/train/p10000032/s01/'
# image = Image.open(os.path.join(path, 'view1_frontal.jpg'))
# image = image.convert('RGB')

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False
)

train_generator = train_datagen.flow_from_directory(
    'data/train_p10/train/p10000032',
    target_size=(299, 299),
    batch_size=32,
    class_mode=None
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False
)

val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(299, 299),
    batch_size=32,
    class_mode=None
)
