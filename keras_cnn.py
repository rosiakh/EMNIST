from __future__ import print_function

import numpy as np
import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import Flatten, Lambda, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam as Adam
from keras.optimizers import RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from model import NeuralNet

class Cnn(NeuralNet):

    def prepare_data(self, dataset):
        """Takes data loaded by spio.loadmat and prepares training and test sets."""

        # load training dataset
        x_train = dataset["dataset"][0][0][0][0][0][0]
        x_train = x_train.astype(np.float32)

        # load training labels
        y_train = dataset["dataset"][0][0][0][0][0][1]

        # load test dataset
        x_test = dataset["dataset"][0][0][1][0][0][0]
        x_test = x_test.astype(np.float32)

        # load test labels
        y_test = dataset["dataset"][0][0][1][0][0][1]

        # store labels for visualization
        #train_labels = y_train
        #test_labels = y_test

        x_train /= 255
        x_test /= 255

        # reshape using matlab order
        #x_train = x_train.reshape(x_train.shape[0], 1, 28, 28, order="A")
        #x_test = x_test.reshape(x_test.shape[0], 1, 28, 28, order="A")

        # for "EMNIST Letters" (they're indexed from 1 instead of 0)

        y_min = y_train.min()
        y_max = y_train.max()

        num_classes = y_max - y_min + 1
        y_train -= y_min
        y_test -= y_min

        #x_train, y_train = self.expand_dataset(x_train, y_train)

        # labels should be onehot encoded
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        img_rows, img_cols = 28, 28

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        return (x_train, y_train), (x_test, y_test), num_classes, input_shape

    def print_dataset_stats(self, x_train, y_train, x_test, y_test):

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

    def create_net(self, num_classes, input_shape):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        return model