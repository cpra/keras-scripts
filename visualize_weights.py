
'''
Visualize weights of a trained network.
~ Christopher
'''

from kutils import utils

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import json
import math

# functions


def make_mosaic(imgs, nrows, ncols, border=1):
    '''
    Adapted from https://github.com/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    '''

    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = np.ma.masked_all((nrows * imshape[0] + (nrows - 1) * border, ncols * imshape[1] + (ncols - 1) * border), dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border

    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0], col * paddedw:col * paddedw + imshape[1]] = imgs[i]

    return mosaic


# parse args

parser = argparse.ArgumentParser(description='Visualize weights of a Keras network that was trained using `train.py`.')
parser.add_argument('netinfo', type=str, help='Path to a `.info` file written by `train.py`')
parser.add_argument('layer', type=int, help='Index of the layer to visualize (0 = first, 1 = second, ...)')
parser.add_argument('--map', type=int, default=0, help='Index of the input feature map')
args = parser.parse_args()

assert(os.path.isfile(args.netinfo))

# load data

print('Loading "{}" ...'.format(args.netinfo))

with open(args.netinfo, 'r') as f:
    netinfo = json.load(f)

# load and compile net

model = utils.load_compile_model(netinfo, args.netinfo, True)

# visualize

layer = model.layers[args.layer]

num_params = 0
if hasattr(layer, 'params'):
    for p in layer.params:
        num_params += p.get_value().size

print('Visualizing weights of layer [{}] {} {} => {} [{} params] ...'.format(args.layer, type(layer).__name__, layer.input_shape[1:], layer.output_shape[1:], num_params))

if num_params > 0:
    W = layer.W.get_value(borrow=True)
    W = np.squeeze(W)

    print(' Shape: {}'.format(W.shape))
    if W.ndim == 4:
        print(' Visualizing input channel {}'.format(args.map))
        W = W[args.map, :, :, :]

    nrows = math.ceil(math.sqrt(W.shape[0]))
    ncols = math.ceil(W.shape[0] / nrows)

    plt.imshow(make_mosaic(W, nrows, ncols), interpolation='none', cmap='hot')
    plt.show()
else:
    print(' Layer has no weights')
