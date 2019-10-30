import os
import pandas as pd
import numpy as np
from my_classes import DataGenerator, PredictDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import pickle
from keras.models import load_model
import keras.backend as K
from keras import optimizers
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight

def getListOfImages(dirName):
    allFiles = list()
    for (dir_path, dir_name, file_name) in os.walk(dirName):
        allFiles += [os.path.join(dir_path, file) for file in file_name]
        #imageList = [remove_cur_dir(f) for f in allFiles]
        # return to this later. Might be better to keep the train or valid part so that we can more easily load with the generator
        imageList = allFiles
    return imageList


def remove_cur_dir(l):
    return l[2:]

# def auroc(y_true, y_pred):
#    if len(np.unique(y_true)) == 1:
#        return 0
#    else:
#        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def auc_roc(y_true, y_pred):
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def mean_label(y_true, y_pred):
    return K.mean(y_true)


load_data = 1
if load_data:
    print('loading model and data')
    os.chdir('/home/steve/PycharmProjects/mimic-cxr/data')
   # os.chdir('/media/steve/Samsung_T5/MIMICCXR')
    with open('partition.pkl', 'rb') as handle:
        partition = pickle.load(handle)
    with open('labels.pkl', 'rb') as handle:
        labels = pickle.load(handle)
    # model = load_model('test_model.h5')
else:
    print('starting')
    # os.chdir('/home/steve/PycharmProjects/mimic-cxr/data')
    os.chdir('/media/steve/Samsung_T5/MIMICCXR')
    trainFiles = getListOfImages('train')
    valFiles = getListOfImages('valid')

    # For now, we are dropping the laterals here
    # Later should be adding them to their own dict for separate model
    trainFiles = [f for f in trainFiles if 'frontal' in f]
    valFiles = [f for f in valFiles if 'frontal' in f]

    # make dictionary with path to all training and validation images
    partition = {
        'train': trainFiles,
        'validation': valFiles
    }

    print('loaded paths')
    # Now we make a dictionary with the labels for each by referencing to the csv file
    labels = {}
    trainingDF = pd.read_csv('train.csv')
    validDF = pd.read_csv('validate.csv')
    for f in trainFiles:
        labels[f] = trainingDF.loc[trainingDF['path'] == f].iloc[:,2:].to_numpy()
    for f in valFiles:
        labels[f] = validDF.loc[validDF['path'] == f].iloc[:,2:].to_numpy()
    with open('partition.pkl', 'wb') as handle:
        pickle.dump(partition, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('labels.pkl', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('loaded labels, saved files')

params = {'dim': (299,299),
          'batch_size': 16,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': False
          }

print('initializing data generator')
finding=7
train_gen = DataGenerator(partition['train'],labels,finding,**params)
val_gen = DataGenerator(partition['validation'],labels,finding,**params)

print('making model')
"""
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(299, 299, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
"""
# Networks
from keras.applications.xception import Xception

# Layers
from keras.layers import *

#Prepare the Model
from keras.applications.xception import preprocess_input
HEIGHT = 299
WIDTH = 299
preprocessing_function = preprocess_input
base_model = Xception(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = True

    # for layer in model.layers[:base_model_last_layer_number]:
    #    layer.trainable = False
    # for layer in model.layers[base_model_last_layer_number:]:
    #    layer.trainable = True

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

class_list = ["normal", "pneumonia"]
FC_LAYERS = [32, 64]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))

# sgd = optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9, nesterov=True)
finetune_model.compile(loss='binary_crossentropy',
                    optimizer='Nadam',
                    metrics=['accuracy', auc_roc])
print('fitting model')
class_weights = {0: 1.,
                 1: 5.}

finetune_model.fit_generator(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        workers=6,
        class_weight=class_weights,
        verbose=1)

print('all done. Now making some predictions')


params = {'dim': (299,299),
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': False
          }
gen = PredictDataGenerator(partition['validation'],**params)

num_to_pred = 200
probs = finetune_model.predict_generator(gen,num_to_pred)

finding = 7
IDs = partition['validation'][0:num_to_pred]
temp_labels = [labels[ID][0][finding] for ID in IDs]
temp_labels = np.asarray(temp_labels)
temp_labels[np.isnan(temp_labels)] = 0
temp_labels[temp_labels==-1] = 1