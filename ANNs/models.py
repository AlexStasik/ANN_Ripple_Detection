import tensorflow as tf
import numpy as np
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import matplotlib as mpl

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model
from keras.layers.normalization import BatchNormalization


def save_model(model, save_dir='models', name=''):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    t = time.strftime("%Y%m%d-%H:%M:%S")
    path = os.path.join(save_dir, t+'_'+str(name)+'.h5')
    model.save(path)


def generate_model_LSTM(input_shape, padding='same'):
    keras.backend.clear_session()

    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.LSTM(20, activation='tanh', return_sequences=True)(inputs)
    x = keras.layers.GaussianNoise(1.5)(x)
    # x = keras.layers.LSTM(100, activation='tanh', return_sequences=True)(x)
    x = keras.layers.LSTM(20, activation='tanh', return_sequences=False)(x)

    # x = keras.layers.Flatten()(x)
    predictions = keras.layers.Dense(2, activation='softmax')(x)

    model = keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['accuracy'],
                  # sample_weight_mode='temporal',
                  )

    return model


def generate_model_CNN(input_shape, padding='same'):
    keras.backend.clear_session()

    input_layer = keras.layers.Input(shape=input_shape)
    # noise_output = keras.layers.GaussianNoise(1.5)(input_layer)

    conv1 = keras.layers.Conv1D(filters=10,kernel_size=7,padding=padding,
                                activation='relu')(input_layer)
    conv1 = keras.layers.AveragePooling1D(pool_size=2)(conv1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.GaussianNoise(1.5)(conv1)


    conv2 = keras.layers.Conv1D(filters=10,kernel_size=7,padding=padding,
                                activation='relu')(conv1)
    conv2 = keras.layers.AveragePooling1D(pool_size=2)(conv2)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.GaussianNoise(1.5)(conv2)

    conv3 = keras.layers.Conv1D(filters=10,kernel_size=7,padding=padding,
                                activation='relu')(conv2)
    conv3 = keras.layers.AveragePooling1D(pool_size=2)(conv3)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.GaussianNoise(1.5)(conv3)

    flatten_layer = keras.layers.Flatten()(conv3)
    flatten_layer = keras.layers.BatchNormalization()(flatten_layer)

    full_conencted1 = keras.layers.Dense(50)(flatten_layer)
    full_conencted1 = keras.layers.BatchNormalization()(full_conencted1)

    full_conencted2 = keras.layers.Dense(50)(full_conencted1)

    output_layer = keras.layers.Dense(units=2,activation='softmax')(full_conencted1)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=['categorical_crossentropy'],
                  optimizer=keras.optimizers.Adam(lr=0.005),
                  metrics=['accuracy'],
                  )

    return model


def plot_performance(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.show()


# def make_accuracy_matrix_plot(model, validate_generator, ref='truth'):
#     X_trial, y_trial, = next(validate_generator)
#     res = model.predict(X_trial)
#
#     print(y_trial)
#     print(res)
#
#     y = decode(y_trial)
#     res = decode(res)
#
#
#     plt.figure()
#     plt.plot(y)
#     plt.plot(res)
#     plt.show()
#
#     res_matrix = np.zeros((2,2))
#     if ref=='truth':
#         for i, true_label in enumerate(set(y)):
#             for j, reco_label in enumerate(set(y)):
#                 mask_true = y==true_label
#                 mask_reco = res==reco_label
#                 res_matrix[i,j] = np.sum(mask_true*mask_reco) / np.sum(mask_true)
#     if ref=='est':
#         for i, true_label in enumerate(set(y)):
#             for j, reco_label in enumerate(set(y)):
#                 mask_true = y==true_label
#                 mask_reco = res==reco_label
#                 res_matrix[i,j] = np.sum(mask_true*mask_reco) / np.sum(mask_reco)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     im = ax.matshow(res_matrix, norm=mpl.colors.Normalize(vmin=0., vmax=1.))
#     for i in range(res_matrix.shape[0]):
#         for j in range(res_matrix.shape[0]):
#             if res_matrix[i, j]>0.5:
#                 color = 'k'
#             else:
#                 color = 'w'
#             text = ax.text(j, i, np.round(res_matrix[i, j], decimals=3),
#             ha='center', va='center', color=color)
#     ax.set_xlabel('CNN class')
#     ax.set_ylabel('manual class')
#     ax.xaxis.set_ticks_position('top')
#     # ax.colorbar(im)
#     plt.savefig('plots/res.pdf')
#     plt.show()
