import numpy as np
import keras
import math
from keras.preprocessing.image import load_img
from PIL import Image

class PriorProbability(keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """
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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, finding, batch_size=32, dim=(299,299), n_channels=3,
                 n_classes=2, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.finding = finding
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # counter = 0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Will count NaNs as 0 will count -1 as 1 for now.
            # Need to re-address this later

            # if 'lateral' in ID:
            #    counter += 1
            #    continue

            image = Image.open(ID)
            image = image.resize((299,299))
            image = image.convert('RGB')
            image = np.array(image)
            image = image/255
            image = (image-mean)/std

            # not flipping right now

            X[i,] = image

            # Store class
            temp_label = self.labels[ID][0][self.finding]
            # print(temp_label)
            if np.isnan(temp_label):
                temp_label = 0
            temp_label = int(temp_label)
            if temp_label==-1:
                temp_label = 1
            y[i] = temp_label

           # X = X[:-1*counter,]
           # y = y[:-1*counter]
        #return X, y
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


class PredictDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=1, dim=(299,299), n_channels=3,
                 n_classes=2, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # counter = 0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Will count NaNs as 0 will count -1 as 1 for now.
            # Need to re-address this later

            # if 'lateral' in ID:
            #    counter += 1
            #    continue

            image = Image.open(ID)
            image = image.resize((299,299))
            image = image.convert('RGB')
            image = np.array(image)
            image = image/255
            image = (image-mean)/std

            # not flipping right now

            X[i,] = image

           # X = X[:-1*counter,]
           # y = y[:-1*counter]
        return X
        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

