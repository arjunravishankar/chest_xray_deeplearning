from keract import get_activations, display_activations, display_heatmaps
import pandas as pd
from PIL import Image
from keras.models import load_model
import numpy as np
import matplotlib
import os
import tensorflow as tf
os.chdir('/home/steve/PycharmProjects/mimic-cxr/data')


def auc_roc(y_true, y_pred):
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

# get the path to the image we want to use:
#[train_df,val_df] = pd.read_hdf('cxr_data.h5')
model = load_model('model.h5', custom_objects={'auc_roc':auc_roc})

#pick arbitrary image for now
#path = val_df.iloc[1,'path']
path='valid/p10382575/s07/view1_frontal.jpg'
image = Image.open(path)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image = image.resize((299,299))
image = image.convert('RGB')
image = np.array(image)
image = image/255
image = (image-mean)/std
image = np.expand_dims(image,axis=0)
#get activations
activations = get_activations(model,image,layer_name='block14_sepconv2')
display_activations(activations)
display_heatmaps(activations,image)

