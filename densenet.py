import os
import pandas as pd
import numpy as np
import skimage
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
from keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import keras_metrics as km
import math
import keras
import initializers

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

"""
class PriorProbability(keras.initializers.Initializer):
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result
"""

#K.set_learning_phase(1)
import logging
LOG = "mimic.log"
logging.basicConfig(filename=LOG, filemode="w", level=logging.ERROR)
# console handler
console = logging.StreamHandler()
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

K.clear_session()

print('starting')
#os.chdir('/home/steve/PycharmProjects/mimic-cxr/data')
os.chdir('/media/steve/Samsung_T5/MIMICCXR')
train_df = pd.read_csv('train.csv')
train_df = train_df[train_df.view.str.contains('frontal')]
finding = 'Cardiomegaly'
print('Formatting labels for finding: ' + finding)
train_df[finding].fillna(0, inplace=True)
n=10000
val_df = train_df.iloc[-1*n:,:]
train_df.drop(train_df.tail(n).index,inplace=True)
train_df=train_df[train_df[finding]!=-1]
train_df[finding]=train_df[finding].astype(str)
val_df=val_df[val_df[finding]!=-1]
val_df[finding]=val_df[finding].astype(str)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
print('Initializing data generators')
class_list = ["0.0", "1.0"]
#directory='/home/steve/PycharmProjects/mimic-cxr/data',
directory='/media/steve/Samsung_T5/MIMICCXR'
train_gen = train_datagen.flow_from_dataframe(dataframe=train_df,directory=directory,x_col='path',y_col=finding,target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=16,shuffle=True,classes=class_list)
val_gen = val_datagen.flow_from_dataframe(dataframe=val_df,directory=directory,x_col='path',y_col=finding,target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=16,shuffle=True,classes=class_list)

print('Making model')
# Networks
from keras.applications.densenet import DenseNet121

# Layers
from keras.layers import *


#class FrozenBatchNormalization(layers.BatchNormalization):
#    def call(self, inputs, training=None):
#        return super().call(inputs=inputs, training=False)


#BatchNormalization = layers.BatchNormalization
#layers.BatchNormalization = FrozenBatchNormalization

#Prepare the Model
HEIGHT = 224
WIDTH = 224
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=2)

#undo the patch
#layers.BatchNormalization = BatchNormalization

"""
for layer in base_model.layers[:115]:
    layer.trainable = False
for layer in base_model.layers[115:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
dropout=0.5
x = Dense(200, activation='relu')(x)
x = Dropout(dropout)(x)
prior_probability=0.05
x = Dense(50, activation='relu',bias_initializer=initializers.PriorProbability(probability=prior_probability))(x)
x = Dropout(dropout)(x)
num_classes = len(class_list)
predictions = Dense(num_classes, activation='softmax')(x)
finetune_model = Model(inputs=base_model.input, outputs=predictions)
"""
### build without prior probability set
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    #for layer in base_model.layers:
    #    layer.trainable = False
    #for layer in base_model.layers[:126]:
    #    layer.trainable=False
    #for layer in base_model.layers[126:]:
    #    layer.trainable=True

    x = base_model.output
    #x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
   # prior_probability = 0.05
   # x = Dense(50, activation='relu', bias_initializer=initializers.PriorProbability(probability=prior_probability))(x)
   # x = Dropout(dropout)(x)
    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

FC_LAYERS = [512, 512, 512, 512]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))
# end model building section

recall = km.binary_recall(label=1)
precision = km.binary_precision(label=1)

finetune_model.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy', auc_roc, km.binary_precision(label=1), km.binary_recall(label=1)])

from sklearn.utils import class_weight
cws = class_weight.compute_class_weight('balanced', np.unique(train_gen.classes),train_gen.classes)

print('fitting model')
class_weights = {0: cws[0],
                 1: cws[1]}

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size
es = EarlyStopping(monitor='val_loss',patience=5,verbose=1,restore_best_weights=True)

history= finetune_model.fit_generator(
        generator=train_gen,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=100,
        callbacks=[es],
        validation_data=val_gen,
        validation_steps=STEP_SIZE_VALID,
        workers=6,
        class_weight=class_weights,
        verbose=1)

print('evaluate phase I model and save to log file')
e = finetune_model.evaluate_generator(val_gen, steps=STEP_SIZE_VALID, verbose=1)
logger.error('Phase I: ' + str(e))
p = finetune_model.predict_generator(val_gen, steps=STEP_SIZE_VALID, verbose=1)
import matplotlib.pyplot as plt
plt.plot(p[:,1])
plt.show()


from keras.preprocessing import image
from keras.applications.densenet import decode_predictions
img_path = '/media/steve/Samsung_T5/MIMICCXR/valid/p10228846/s01/view1_frontal.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x = preprocess_input(x)
preds = finetune_model.predict(x)
tumor = finetune_model.output[:,1]
last_conv_layer = finetune_model.get_layer('conv5_block16_2_conv') #'conv5_block16_2_conv'
grads = K.gradients(tumor, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([finetune_model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap *= -1
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
#plt.matshow(heatmap)
import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255*heatmap)
#heatmap = 255-heatmap
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#superimposed_img = heatmap* 0.4 + bw_img

from skimage import data, color, io, img_as_float

# Convert the input image and color mask to Hue Saturation Value (HSV)
# colorspace
img_hsv = color.rgb2hsv(img)
color_mask_hsv = color.rgb2hsv(heatmap)

# Replace the hue and saturation of the original image
# with that of the color mask
img_hsv[..., 0] = color_mask_hsv[..., 0]
img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6

img_masked = color.hsv2rgb(img_hsv)

#cv2.imwrite('/home/steve/PycharmProjects/hyperfine/heatmap.jpg',img_masked)
plt.imshow(img_masked)
#plt.savefig('/home/steve/PycharmProjects/hyperfine/heatmap.jpg', bbox_inches='tight', dpi=300)
plt.show()


#generate ROC curves
auc_gen = val_datagen.flow_from_dataframe(dataframe=val_df,directory=directory,x_col='path',y_col=finding,target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=1,shuffle=False,classes=class_list)
auc_preds = finetune_model.predict_generator(auc_gen, steps=auc_gen.n, verbose=1)
lab = auc_gen.classes
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(lab, auc_preds[:,1])
AUC = auc(fpr, tpr)
plt.plot(fpr,tpr,color='0.5')
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('/home/steve/PycharmProjects/mimic-cxr/HF_ROC.png',bbox_inches='tight', dpi=300)
plt.show()