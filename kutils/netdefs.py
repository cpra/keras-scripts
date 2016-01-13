
'''
General network definitions for Keras, for use with `train.py`.
~ Christopher
'''


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D


def example_cifar10_cnn(channels, rows, cols, classes):
    '''
    Return the network that is used in Keras' CIFAR10 example,
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py.

    Args:
        channels: number of image channels
        rows: number of image rows
        cols: number of image columns
        classes: number of classes
    Returns:
        network model
    '''

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(channels, rows, cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model
