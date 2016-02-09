
'''
Average predictions saved using `test.py`.
~ Christopher
'''

from kutils import utils

import numpy as np

import os
import argparse


# parse args

parser = argparse.ArgumentParser(description='Average predictions saved using `test.py`.')
parser.add_argument('gt', type=str, help='Path to a HDF5 database that contains the the original data, `X` ((n*c*h*w) numpy array of image data) and `y` ((n,) numpy array of labels)')
parser.add_argument('predictions', type=str, nargs='+', help='Paths to prediction files')
args = parser.parse_args()

assert(os.path.isfile(args.gt))
assert(all([os.path.isfile(p) for p in args.predictions]))

# load data

_, y = utils.load_h5_db(args.gt, True)

y = np.ravel(y)

probas = []
for p in args.predictions:
    probas.append(np.load(p))

# average

probas = np.array(probas)
probas = np.average(probas, axis=0)

cls = np.argmax(probas, axis=1)
assert(y.size == cls.size)

# compute ensemble accuracy

print('Accuracy: {:.3f}%'.format(100.0 * np.sum(cls == y) / y.size))

for c in np.unique(y):
    is_class_true = y == c
    is_class_pred = cls == c
    is_correct = np.logical_and(is_class_true, is_class_pred)
    print(' Class {}: {} true samples, {} correctly predicted'.format(c, np.sum(is_class_true), np.sum(is_correct)))
