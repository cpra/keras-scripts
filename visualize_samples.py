
'''
Apply a trained Keras net on test data.
~ Christopher
'''

from kutils import utils

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse
import json

# parse args

parser = argparse.ArgumentParser(description='Test a Keras network that was trained using `train.py`.')
parser.add_argument('netinfo', type=str, help='Path to a `.info` file written by `train.py`')
parser.add_argument('data', type=str, help='Path to a HDF5 database that contains the test data, `X` ((n*c*h*w) numpy array of image data) and `y` ((n,) numpy array of labels)')
parser.add_argument('--augment', type=str, help='Which data augmentation to apply (default = none, "train" = training augmentation, "val" = validation/test augmentation)')
parser.add_argument('--shuffle', action='store_true', help='Shuffle minibatches')
args = parser.parse_args()

assert(os.path.isfile(args.netinfo))
assert(os.path.isfile(args.data))
assert(args.augment in (None, 'train', 'val'))

# load data

print('Loading "{}" ...'.format(args.netinfo))

with open(args.netinfo, 'r') as f:
    netinfo = json.load(f)

print('Loading training settings "{}" ...'.format(netinfo['train_args']['fpath']))

with open(netinfo['train_args']['fpath'], 'r') as f:
    traininfo = json.load(f)

X, y = utils.load_h5_db(args.data, True)
X = X.astype(np.float32)

if netinfo['preprocess']['demean']:
    for i, m in enumerate(netinfo['preprocess']['channel_means']):
        X[:, i, :, :] -= m

if netinfo['preprocess']['divide'] != 1:
    X /= netinfo['preprocess']['divide']

# show

batchgen = utils.BatchGen(X, y, 16, True if args.shuffle else False, netinfo['num_classes'])

if args.augment:
    print('Applying "{}" augmentations "{}"'.format(args.augment, ','.join([p for p in traininfo['augment'][args.augment]])))
    batchgen = utils.MinibatchProcessor(batchgen, traininfo['augment'][args.augment])

for X_batch, Y_batch in batchgen:
    for s in range(X_batch.shape[0]):
        sample = np.rollaxis(X_batch[s, :, :, :], 0, 3)
        if sample.shape[2] == 1:
            sample = sample[:, :, 0]

        print('Class: {}'.format(np.argmax(Y_batch[s, :])))

        plt.imshow(sample, interpolation='none')
        plt.colorbar()
        plt.show()
