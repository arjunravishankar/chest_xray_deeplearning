import pandas as pd
import os
import tarfile
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
import logging
from keras.applications.xception import Xception
from keras.layers import *
import keras_metrics as km
from keras import layers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.models import Model
import tensorflow as tf


def extract_tar():
    """
    extract_tar extracts images from tar files
    :return: extracted files in images folder
    """
    os.chdir('/media/steve/Samsung_T5/MIMICCXR/chest_xray_14')

    for n in range(1, 13):
        file = 'images_' + str(n).zfill(2) + '.tar.gz'
        print('Starting ' + file)
        tar = tarfile.open(file)
        tar.extractall()
        tar.close()
        print('Extracted ' + file)

    print('Finished file extraction.')


def create_dfs():
    """
    create_dfs creates training validation dataframe of image files and labels
    :return:
    """
    os.chdir('/media/steve/Samsung_T5/MIMICCXR/chest_xray_14')
    chx14_df = pd.read_csv('Data_Entry_2017.csv')
    unique_labels = chx14_df['Finding Labels'].str.split('|', expand=True)[0].unique()
    for label in unique_labels:
        chx14_df[label] = chx14_df['Finding Labels'].str.contains(pat=label)
        chx14_df[label] = np.where(chx14_df[label], '1.0', '0.0')

    # Rename 'Image Index' column to 'path'
    chx14_df.rename(index=str, columns={'Image Index': 'path'}, inplace=True)

    # Create training Dataframe
    train_df = pd.read_csv('train_val_list.txt', sep="\n", header=None)
    train_df = train_df.merge(chx14_df, left_on=[0], right_on='path')

    # Split Training Data
    n = 10000
    val_df = train_df.iloc[-1 * n:, :]
    train_df.drop(train_df.tail(n).index, inplace=True)
    print('Created train and val df')
    return train_df, val_df


def initialize_datagen(train_df, val_df, finding, class_list):
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=True)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(dataframe=train_df,
                                                  directory='/media/steve/Samsung_T5/MIMICCXR/chest_xray_14/images',
                                                  x_col='path', y_col=finding, target_size=(299, 299), color_mode='rgb',
                                                  class_mode='categorical', batch_size=16, shuffle=True,
                                                  classes=class_list)
    val_gen = val_datagen.flow_from_dataframe(dataframe=val_df,
                                              directory='/media/steve/Samsung_T5/MIMICCXR/chest_xray_14/images',
                                              x_col='path', y_col=finding, target_size=(299, 299), color_mode='rgb',
                                              class_mode='categorical', batch_size=16, shuffle=True, classes=class_list)
    print('initialized data generators')

    return train_gen, val_gen


def create_log():
    LOG = "CHX_14_eval_cm.log"
    logging.basicConfig(filename=LOG, filemode="w", level=logging.ERROR)
    # console handler
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)
    logger = logging.getLogger(__name__)
    return logger


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    # for layer in base_model.layers:
    #    layer.trainable = False
    for layer in base_model.layers[:126]:
        layer.trainable = False
    for layer in base_model.layers[126:]:
        layer.trainable = True

    x = base_model.output
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model


def auc_roc(y_true, y_pred):
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


class FrozenBatchNormalization(layers.BatchNormalization):
    def call(self, inputs, training=None):
        return super().call(inputs=inputs, training=False)


# main()

# Arguments
finding = 'Pneumonia'
class_list = ['0.0', '1.0']
FC_LAYERS = []
dropout = 0.5
class_weights = {0: 1.,
                 1: 1.}

# Create Data Generators
train_df, val_df = create_dfs()
train_gen, val_gen = initialize_datagen(train_df, val_df, finding, class_list)
logger = create_log()

# Prepare the model
BatchNormalization = layers.BatchNormalization
layers.BatchNormalization = FrozenBatchNormalization
HEIGHT = 299
WIDTH = 299
base_model = Xception(layers=layers, weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

# Undo the patch
layers.BatchNormalization = BatchNormalization

# Build the model
finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))
# End model building section

# Define Error Metrics
recall = km.binary_recall(label=1)
precision = km.binary_precision(label=1)

# Compile the model
finetune_model.compile(loss='binary_crossentropy',
                       optimizer='nadam',
                       metrics=['accuracy', auc_roc, km.binary_precision(label=1), km.binary_recall(label=1)])


STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = val_gen.n // val_gen.batch_size
es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

history_prelim = finetune_model.fit_generator(
    generator=train_gen,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=50,  # make sure to change back
    callbacks=[es],
    validation_data=val_gen,
    validation_steps=STEP_SIZE_VALID,
    workers=6,
    class_weight=class_weights,
    verbose=1)

print('Evaluate Model and save to log file')
e_phaseI = finetune_model.evaluate_generator(val_gen, steps=STEP_SIZE_VALID, verbose=1)
logger.error('Evaluation on balanced set: ' + str(e_phaseI))
p = finetune_model.predict_generator(val_gen, steps=STEP_SIZE_VALID, verbose=1)
plt.plot(p)
plt.show()

# def main():

# if __name__ == '__main__':
#    chx14_df = main()
