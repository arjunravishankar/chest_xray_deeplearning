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
from keras import optimizers, applications, layers
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import keras_metrics as km

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


def old_auc(y_true, y_pred):
    #auc = tf.metrics.auc(y_true, y_pred)[1]
    auc = tf.compat.v1.metrics.auc(y_true, y_pred, num_thresholds=400)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def auc_roc(y_true, y_pred):
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

#def new_auc(y_true, y_pred) :
#    if len(np.unique(y_true)) == 1:
#        return 0
#    score = tf.py_func( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
#                        [y_true, y_pred],
#                        'float32',
#                        stateful=False,
#                        name='sklearnAUC' )
#    return score

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def mean_label(y_true, y_pred):
    return K.mean(y_true)

dependencies = {
    'FrozenBatchNormalization':FrozenBatchNormalization,
    'auc_roc': auc_roc,
    'binary_precision': km.binary_precision(label=1),
    'binary_recall': km.binary_recall(label=1)
}
model = load_model('model.h5', custom_objects=dependencies)
print('loaded')
for layer in model.layers[:135]:
    layer.trainable=False
for layer in model.layers[135:]:
    layer.trainable=True
class_weights = {0: 1.,
                 1: 10.}

print('about to compile')
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
              loss=['binary_crossentropy'],
              metrics=['accuracy', auc_roc, precision, recall])
print('compiling done, about to train')
#directory='/home/steve/PycharmProjects/mimic-cxr/data',
#directory='/media/steve/Samsung_T5/MIMICCXR',
phaseII_train_gen = train_datagen.flow_from_dataframe(dataframe=train_df,directory='/media/steve/Samsung_T5/MIMICCXR',x_col='path',y_col=finding,target_size=(299,299),color_mode='rgb',class_mode='categorical',batch_size=16,shuffle=True,classes=class_list)
phaseII_val_gen = val_datagen.flow_from_dataframe(dataframe=val_df,directory='/media/steve/Samsung_T5/MIMICCXR',x_col='path',y_col=finding,target_size=(299,299),color_mode='rgb',class_mode='categorical',batch_size=16,shuffle=False,classes=class_list)
STEP_SIZE_TRAIN=phaseII_train_gen.n//phaseII_train_gen.batch_size
STEP_SIZE_VALID=phaseII_val_gen.n//phaseII_val_gen.batch_size

history = model.fit_generator(
        generator=phaseII_train_gen,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=10,    # make sure to change back
        callbacks=[es],
        validation_data=phaseII_val_gen,
        validation_steps=STEP_SIZE_VALID,
        workers=6,
        class_weight=class_weights,
        verbose=1)

print('all done')

# get testing results on the same validation data
e = model.evaluate_generator(phaseII_val_gen, steps=STEP_SIZE_VALID, verbose=1)
p = model.predict_generator(phaseII_val_gen, steps=STEP_SIZE_VALID, verbose=1)
import matplotlib.pyplot as plt
plt.plot(p)
plt.show()