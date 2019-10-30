from my_classes import PredictDataGenerator
import numpy as np
import pickle
from keras.models import load_model
import os

os.chdir('/home/steve/PycharmProjects/mimic-cxr/data')
with open('partition.pkl', 'rb') as handle:
    partition = pickle.load(handle)
with open('labels.pkl', 'rb') as handle:
    labels = pickle.load(handle)
model = load_model('test_model.h5')

params = {'dim': (299,299),
          'batch_size': 1,
          'n_classes': 2,
          'n_channels': 3,
          'shuffle': False
          }
gen = PredictDataGenerator(partition['validation'],**params)

num_to_pred = 10
probs = model.predict_generator(gen,num_to_pred)

finding = 7
IDs = partition['validation'][0:num_to_pred]
temp_labels = [labels[ID][0][finding] for ID in IDs]
temp_labels = np.asarray(temp_labels)
temp_labels[np.isnan(temp_labels)] = 0
temp_labels[temp_labels==-1] = 1