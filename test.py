
'''
Apply a trained Keras net on test data.
~ Christopher
'''

from kutils import utils

import numpy as np

import os
import argparse
import json

# parse args

parser = argparse.ArgumentParser(description='Test a Keras network that was trained using `train.py`.')
parser.add_argument('netinfo', type=str, help='Path to a `.info` file written by `train.py`')
parser.add_argument('data', type=str, help='Path to a HDF5 database that contains the test data, `X` ((n*c*h*w) numpy array of image data) and `y` ((n,) numpy array of labels)')
parser.add_argument('--save_results', type=str, help='Save results (predicted probabilities) to a npy file with the specified name')
parser.add_argument('--batchsize', type=int, default=64, help='Minibatch size')
args = parser.parse_args()

assert(os.path.isfile(args.netinfo))
assert(os.path.isfile(args.data))
assert(not args.save_results or not os.path.exists(args.save_results))

# load data

print('Loading "{}" ...'.format(args.netinfo))

with open(args.netinfo, 'r') as f:
    netinfo = json.load(f)

X, y = utils.load_h5_db(args.data, True)
X = X.astype(np.float32)

if netinfo['preprocess']['demean']:
    for i, m in enumerate(netinfo['preprocess']['channel_means']):
        X[:, i, :, :] -= m

X /= netinfo['preprocess']['divide']

# load and compile net

model = utils.load_compile_model(netinfo, args.netinfo, True)

# test

print('Testing ...')

probas = np.zeros((y.size, netinfo['num_classes']), dtype=np.float32)
results = np.zeros(y.shape, dtype=np.int16) - 1

batchgen = utils.BatchGen(X, y, args.batchsize, False, netinfo['num_classes'])

b = 0
idx0 = 0

for X_batch, Y_batch in batchgen:
    bs = X_batch.shape[0]
    pred = model.predict(X_batch, bs)

    probas[idx0:idx0+bs, :] = pred
    results[idx0:idx0+bs] = np.argmax(pred, axis=1)

    acc = np.sum(results[idx0:idx0+bs] == y[idx0:idx0+bs]) / bs
    print(' batch={}, idx0={}, idx1={}, acc={:.1f}%'.format(b+1, idx0, idx0+bs, acc*100))

    b += 1
    idx0 += bs

assert(np.sum(results < 0) == 0)
print('Accuracy: {:.3f}%'.format(100.0 * np.sum(results == y) / y.size))

for c in np.arange(netinfo['num_classes']):
    is_class_true = y == c
    is_class_pred = results == c
    is_correct = np.logical_and(is_class_true, is_class_pred)
    print(' Class {}: {} true samples, {} correctly predicted'.format(c, np.sum(is_class_true), np.sum(is_correct)))

if args.save_results:
    np.save(args.save_results, probas)
