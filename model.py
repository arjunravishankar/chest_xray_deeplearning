#image imports
from skimage import io, transform
from PIL import Image
#general
import os
import time
import pandas as pd
import numpy as np

import make_dataset as dset

#keras imports
import keras
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions

def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay):
    """
    :param model: keras model to be finetuned
    :param criterion: loss criterion
    :param optimizer: optimizer for training (SGD)
    :param LR: learning rate
    :param num_epochs: max num training epochs
    :param dataloaders: keras dataloaders
    :param dataset_sizes: size of training and validation sets
    :param weight_decay: weight decay parameter
    :return: model - trained keras model
            best_epoch - epoch with min loss
    """

    xception_mdl = Xception(weights='imagenet')


    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # transfer learning, set up network

    #custom datagenerator for fine tuning