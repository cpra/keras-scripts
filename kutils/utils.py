
'''
Misc utility functions.
~ Christopher
'''

from keras import optimizers as keras_optimizers
from keras.models import model_from_json
from keras.utils import np_utils as keras_np_utils

import h5py
import numpy as np

import os
import math
import sys
import json
import importlib


def load_h5_db(path, verbose=True):
    '''
    Load feature vectors `X` and class labels `y` from a HDF5 file.

    Args:
        path: path to the file that contains `X` and `y`.
        verbose: print infos to the console.
    Returns:
        tuple (X, y)
    '''

    h5f = h5py.File(path, 'r')

    assert('X' in h5f)
    assert('y' in h5f)

    X = h5f['X'][:]
    y = h5f['y'][:]

    h5f.close()

    if verbose:
        print('Loaded "{}"'.format(os.path.basename(path)))
        print(' {} samples of size {}'.format(X.shape[0], X.shape[1:]))

        uc, cv = np.unique(y, return_counts=True)
        for u, c in zip(uc, cv):
            print('  Class {} : {} samples'.format(u, c))

    return X, y


class BatchGen:

    '''
    Minibatch generator.
    '''

    @staticmethod
    def from_file(fpath, size, shuffle, classes=0):
        '''
        Load features and labels from a HDF5 file and pass everything to the ctor.

        Args:
            fpath: path to a HDF5 file supported by `load_h5_db`.
            size: minibatch size.
            shuffle: whether to shuffle the data.
            classes: number of classes (0 = infer from data).
        '''

        X, y = load_h5_db(fpath, False)
        return BatchGen(X, y, size, shuffle, classes)

    def __init__(self, X, y, size, shuffle, classes=0):
        '''
        Ctor.

        Args:
            self: current instance.
            X: n*d numpy array with n being the number of samples and d being the feature dimensionality.
            y: y: n, numpy array of class labels.
            size: minibatch size.
            shuffle: whether to shuffle the data.
            classes: number of classes (0 = infer from data).
        '''

        self._X = X
        self._Y = keras_np_utils.to_categorical(y, np.unique(y).size if not classes else classes)
        self._size = size

        self._idx = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(self._idx)

    def __len__(self):
        '''
        Support for `len(batchgen)`.
        '''

        return int(math.ceil(self._X.shape[0] / self._size))

    def __iter__(self):
        '''
        Support for `for X_batch, Y_batch in batchgen`.
        '''

        n = len(self)
        for b in range(n):
            idx0 = b * self._size
            idx1 = min(idx0 + self._size, self._X.shape[0])

            if idx0 < idx1:
                X_batch = self._X[self._idx[idx0:idx1], :, :, :].astype(np.float32)
                Y_batch = self._Y[self._idx[idx0:idx1], :]

                yield X_batch, Y_batch


def load_compile_model_for_training(props, num_classes, weights=None, verbose=True):
    '''
    Load and compile a Keras model for training and return it.

    Args:
        props: `dict` obtained by loading training or info JSON files (see `train.py`).
        num_classes: number of classes.
        weights: model weights to load.
        verbose: print infos to the console.
    '''

    mod_name = props['netgen']['path'][:props['netgen']['path'].rfind('.')]
    net_name = props['netgen']['path'][props['netgen']['path'].rfind('.')+1:]

    if verbose:
        print('Loading net using "{}" from module "{}" ...'.format(net_name, mod_name))

    if props['netgen']['spath'] and props['netgen']['spath'] not in sys.path:
        sys.path.insert(0, props['netgen']['spath'])

    net_mod = importlib.import_module(mod_name)
    net_gen = getattr(net_mod, net_name)

    net_gen_args = props['netgen']['args']
    net_gen_args['classes'] = num_classes

    model = net_gen(**net_gen_args)

    if weights:
        model.load_weights(weights)
        if verbose:
            print('Loaded weights from "{}"'.format(weights))

    # compile

    loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

    if verbose:
        print('Compiling model using "{}" loss and "{}" optimizer ...'.format(loss, props['train']['optimizer']['type']))

    opt_ = getattr(keras_optimizers, props['train']['optimizer']['type'])
    opt = opt_(**props['train']['optimizer']['args'])

    model.compile(loss=loss, optimizer=opt)

    if verbose:
        print(' Structure:')
        for l in model.layers:
            print(' ', type(l).__name__, l.input_shape, l.output_shape)

    return model


def load_compile_model(info, info_filepath, verbose=True):
    '''
    Load and compile a Keras model obtained using `train.py` and return it.

    Args:
        info: deserialized JSON info file written by `train.py`.
        info_filepath: path to the JSON file from which `info` was serialized.
        verbose: print infos to the console.
    '''

    # load

    print('Loading model ...')

    model = model_from_json(json.dumps(info['model']))

    wfp = info['weights']
    if not os.path.isabs(wfp):
        wfp = os.path.join(os.path.dirname(info_filepath), wfp)

    assert(os.path.isfile(wfp))

    model.load_weights(wfp)

    if verbose:
        print('Loaded weights from "{}"'.format(wfp))

    # compile

    if verbose:
        print('Compiling model ...')

    model.compile(loss=info['model']['loss'], optimizer='SGD')

    if verbose:
        print(' Structure:')
        for l in model.layers:
            print(' ', type(l).__name__, l.input_shape, l.output_shape)

    return model
