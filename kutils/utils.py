
'''
Misc utility functions.
~ Christopher
'''

from keras import optimizers as keras_optimizers
from keras.models import model_from_json
from keras.utils import np_utils as keras_np_utils

import h5py
import numpy as np

try:
    import cv2
except ImportError as e:
    print('Failed to import cv2 module: "{}". Some functionality will not be available'.format(e))
    cv2 = None

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

    if verbose:
        nparams = 0
        print(' Structure:')
        for l in model.layers:
            np = 0
            if hasattr(l, 'params'):
                for p in l.params:
                    np += p.get_value().size

            print('  {} {} => {} [{} params]'.format(type(l).__name__, l.input_shape[1:], l.output_shape[1:], np))
            nparams += np

        print(' Total number of params: {}'.format(nparams))

    # compile

    loss = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'

    if verbose:
        print('Compiling model using "{}" loss and "{}" optimizer ...'.format(loss, props['train']['optimizer']['type']))

    opt_ = getattr(keras_optimizers, props['train']['optimizer']['type'])
    opt = opt_(**props['train']['optimizer']['args'])

    model.compile(loss=loss, optimizer=opt)

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

    if verbose:
        nparams = 0
        print(' Structure:')
        for l in model.layers:
            np = 0
            if hasattr(l, 'params'):
                for p in l.params:
                    np += p.get_value().size

            print('  {} {} => {} [{} params]'.format(type(l).__name__, l.input_shape[1:], l.output_shape[1:], np))
            nparams += np

        print(' Total number of params: {}'.format(nparams))

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

    return model


class SlidingWindowPatchExtractor:

    '''
    Extract patches from an image in a sliding window fashion.
    '''

    def __init__(self, im, patchsz, offset):
        '''
        Ctor.

        Args:
            self: current instance.
            image: image as a numpy array, h*w or h*w*c
            patchsz: patch size (int or tuple `(height, width)`)
            offset: offset (int or tuple `(vertical, horizontal)`)
        '''

        self.im = im.copy()
        self.patchsz = (patchsz, patchsz) if type(patchsz) is int else patchsz
        self.offset = (offset, offset) if type(offset) is int else offset

    def __iter__(self):
        '''
        Support for `for y0, x0, patch in extractor`.
        '''

        h, w = self.im.shape[0], self.im.shape[1]

        x0 = 0
        while x0+self.patchsz[1] <= w:
            y0 = 0
            while y0+self.patchsz[0] <= h:
                yield (y0, x0, self.im[y0:y0+self.patchsz[0], x0:x0+self.patchsz[1], :])
                y0 += self.offset[0]
            x0 += self.offset[1]


class PatchExtractorMinibatchWrapper:

    '''
    Wrap a patch extractor and convert patches to minibatches.
    '''

    def __init__(self, patch_extractor, batchsz):
        '''
        Ctor.

        Args:
            self: current instance.
            patch_extractor: patch extractor to wrap.
            batchsz: minibatch size.
        '''

        self._ex = patch_extractor
        self._sz = batchsz

    def __iter__(self):
        '''
        Support for `X, coords in extractor`.
        `X` is a n*c*r*w numpy float32 array with 0 < n < batchsz.
        `coords` is a n*2 numpy int array with `coords[i,:]` being the y0 and x0 patch coords of the sample as returned by the patch extractor.
        '''

        ims = self._ex.im.shape
        X = np.empty((self._sz, ims[2] if len(ims) > 2 else 1, self._ex.patchsz[0], self._ex.patchsz[1]), dtype=np.float32)
        coords = np.empty((self._sz, 2), dtype=np.int32)

        i = 0
        n = 0
        for y0, x0, patch in self._ex:
            n += 1
            for c in np.arange(patch.shape[2] if patch.ndim > 2 else 1):
                X[i, c, :, :] = patch[:, :, c]

            coords[i, :] = (y0, x0)

            i += 1

            if i == self._sz:
                yield X, coords
                i = 0

        if i > 0:
            X = X[:i, :, :, :]
            coords = coords[:i, :]
            yield X, coords


class MinibatchProcessor:

    '''
    Apply various modifications to minibatches.
    '''

    class HMirror:

        def __init__(self, proba):
            self.proba = proba

        def process(self, sample):
            return np.fliplr(sample) if np.random.uniform() < self.proba else sample

    class Scale:

        def __init__(self, rows, cols):
            self.rows = rows
            self.cols = cols

        def process(self, sample):
            proc = cv2.resize(sample, (self.cols, self.rows))
            if proc.ndim == 2:
                proc = proc[:, :, np.newaxis]
            return proc

    class RScale:

        def __init__(self, smin, smax):
            assert(smax > smin)

            self.smin = smin
            self.smax = smax

        def process(self, sample):
            s = np.random.uniform(self.smin, self.smax)
            proc = cv2.resize(sample, (0, 0), fx=s, fy=s)
            if proc.ndim == 2:
                proc = proc[:, :, np.newaxis]
            return proc

    class RSim:

        def __init__(self, smin, smax, rmin, rmax, tmin, tmax):
            self.smin = smin
            self.smax = smax
            self.rmin = rmin
            self.rmax = rmax
            self.tmin = tmin
            self.tmax = tmax

        def process(self, sample):
            s = np.random.uniform(self.smin, self.smax)
            r = np.random.uniform(self.rmin, self.rmax)
            t = np.random.randint(self.tmin, self.tmax + 1)

            sy, sx = sample.shape[:2]

            c = (sx / 2 + t, sy / 2 + t)
            sy = int(sy * s)
            sx = int(sx * s)

            mat = cv2.getRotationMatrix2D(c, r, s)

            sample = cv2.warpAffine(sample, mat, (sx, sy), borderMode=cv2.BORDER_REPLICATE)
            if sample.ndim == 2:
                sample = sample[:, :, np.newaxis]

            return sample

    class Crop:

        def __init__(self, rows, cols, location):
            assert(location in ('random', 'center', 'tl', 'tr', 'bl', 'br'))

            self.rows = rows
            self.cols = cols
            self.location = location

        def process(self, sample):
            rows, cols = sample.shape[:2]

            if rows < self.rows or cols < self.cols:
                py = self.rows - rows
                px = self.cols - cols

                pl = 0 if px <= 0 else px // 2
                pr = 0 if px <= 0 else px - pl

                pt = 0 if py <= 0 else py // 2
                pb = 0 if py <= 0 else py - pt

                sample = cv2.copyMakeBorder(sample, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
                if sample.ndim == 2:
                    sample = sample[:, :, np.newaxis]

            if self.location == 'random':
                y0 = np.random.randint(0, max(1, rows - self.rows))
                x0 = np.random.randint(0, max(1, cols - self.cols))
            elif self.location == 'center':
                y0 = (rows-self.rows) // 2
                x0 = (cols-self.cols) // 2
            elif self.location == 'tl':
                y0 = 0
                x0 = 0
            elif self.location == 'tr':
                y0 = 0
                x0 = cols - self.cols
            elif self.location == 'bl':
                y0 = rows - self.rows
                x0 = 0
            else:
                y0 = rows - self.rows
                x0 = cols - self.cols

            return sample[y0:y0+self.rows, x0:x0+self.cols, :]

    _available_processors = ('hmirror', 'scale', 'rscale', 'rsim', 'crop')

    def __init__(self, batchgen, cfg):
        '''
        Ctor.

        Args:
            self: current instance.
            batchgen: minibatch generator to wrap.
            cfg: `dict` with keys being modifications to apply and values being another `dict` of arguments.

        The following keys and arguments are allowed in `cfg` and will be applied in the specified order:
            - `hmirror`: horizontal mirroring
                    - `proba` (float): mirroring probability (0 = never, 1 = always)
            - `scale`: scalling to a fixed size with bilinear interpolation
                    - `rows` (int): number of rows
                    - `cols` (int): number of cols
            - `rscale`: random scaling with bilinear interpolation (requires `ccrop` or `rcrop`)
                    - `smin` (float): minimum scale factor
                    - `smax` (float): maximum scale factor
            - `rsim`: random similarity transforms with border replication and bilinear interpolation (requires `ccrop` or `rcrop`)
                    - `smin` (float): minimum scale factor
                    - `smax` (float): maximum scale factor
                    - `rmin` (float): minimum rotation in degrees
                    - `rmax` (float): maximum rotation in degrees
                    - `tmin` (float): minimum offset of the rotation center from the image center
                    - `tmax` (float): maximum offset of the rotation center from the image center
            - `crop`: center or random crops with automatic border replication if the input is too small
                    - `rows` (int): number of rows of crops
                    - `cols` (int): number of cols of crops
                    - `location` (str): crop location, can be `random`, `center`, `tl` (top left), `tr`, `bl`, `br`
        '''

        self._batchgen = batchgen
        self._cfg = cfg
        self._processors = []

        for p in cfg:
            if p not in MinibatchProcessor._available_processors:
                raise ValueError('Unrecognized processor "{}"'.format(p))

        if 'hmirror' in cfg:
            self._processors.append(MinibatchProcessor.HMirror(**cfg['hmirror']))

        if 'scale' in cfg:
            self._processors.append(MinibatchProcessor.Scale(**cfg['scale']))

        if 'rscale' in cfg:
            self._processors.append(MinibatchProcessor.RScale(**cfg['rscale']))

        if 'rsim' in cfg:
            self._processors.append(MinibatchProcessor.RSim(**cfg['rsim']))

        if 'crop' in cfg:
            self._processors.append(MinibatchProcessor.Crop(**cfg['crop']))

    def __len__(self):
        '''
        Support for `len(processor)`.
        '''

        return len(self._batchgen)

    def __iter__(self):
        '''
        Support for `for X_batch, Y_batch in processor`.
        '''

        for X_batch, Y_batch in self._batchgen:
            samples, channels, rows, cols = X_batch.shape

            if 'scale' in self._cfg:
                rows = self._cfg['scale']['rows']
                cols = self._cfg['scale']['cols']

            if 'crop' in self._cfg:
                rows = self._cfg['crop']['rows']
                cols = self._cfg['crop']['cols']

            X_ret = np.empty((samples, channels, rows, cols), dtype=X_batch.dtype)

            for s in range(X_batch.shape[0]):
                sample = np.rollaxis(X_batch[s, :, :, :], 0, 3)

                for proc in self._processors:
                    sample = proc.process(sample)

                X_ret[s, :, :, :] = np.rollaxis(sample, 2)

            yield X_ret, Y_batch


class MinibatchAugmentor:

    '''
    Similar to to `MinibatchProcessor`, but this class is used to create multiple
    modified versions of each sample, such as center and border crops. The corresponding
    predictions must then be averaged. For this purpose, minibatches created by this class
    come with an additional index vector that encodes which samples belong to the same original one.

    This class supports the same processors as `MinibatchProcessor`, and modified version are
    generated by using these processors (those involving randomness) multiple times per sample.
    '''

    def __init__(self, batchgen, cfg, num, crop5=None):
        '''
        Ctor.

        Args:
            self: current instance.
            batchgen: minibatch generator to wrap.
            cfg: see `MinibatchProcessor` documentation.
            num: number of times to apply the processor pipeline per sample. the number of samples (and minibatch size) is increased by this factor.
            crop5: generate 5 cropped versions (corners and center) for each sample, of the size (rows, cols). this is multiplicative with `num` and will disable `crop` processors.
        '''

        self._batchgen = batchgen
        self._cfg = cfg
        self._num = num
        self._crop5 = crop5
        self._processors = []

        for p in cfg:
            if p not in MinibatchProcessor._available_processors:
                raise ValueError('Unrecognized processor "{}"'.format(p))

        if 'hmirror' in cfg:
            self._processors.append(MinibatchProcessor.HMirror(**cfg['hmirror']))

        if 'scale' in cfg:
            self._processors.append(MinibatchProcessor.Scale(**cfg['scale']))

        if 'rscale' in cfg:
            self._processors.append(MinibatchProcessor.RScale(**cfg['rscale']))

        if 'rsim' in cfg:
            self._processors.append(MinibatchProcessor.RSim(**cfg['rsim']))

        if 'crop' in cfg and crop5 is None:
            self._processors.append(MinibatchProcessor.Crop(**cfg['crop']))

    def __len__(self):
        '''
        Support for `len(augmentor)`.
        '''

        return len(self._batchgen)

    def __iter__(self):
        '''
        Support for `for X_batch, Y_batch, S in augmentor`.
        S is an int numpy array holding indices encoding which samples belong to which original one.
        '''

        bs = None

        if self._crop5 is not None:
            r, c = self._crop5
            c5p = (
                MinibatchProcessor.Crop(r, c, 'center'),
                MinibatchProcessor.Crop(r, c, 'tl'),
                MinibatchProcessor.Crop(r, c, 'tr'),
                MinibatchProcessor.Crop(r, c, 'bl'),
                MinibatchProcessor.Crop(r, c, 'br')
            )
        else:
            c5p = None

        for X_batch, Y_batch in self._batchgen:
            samples, channels, rows, cols = X_batch.shape

            bs = samples * self._num
            if self._crop5 is not None:
                bs *= 5

            if 'scale' in self._cfg:
                rows = self._cfg['scale']['rows']
                cols = self._cfg['scale']['cols']

            if 'crop' in self._cfg:
                rows = self._cfg['crop']['rows']
                cols = self._cfg['crop']['cols']

            if self._crop5 is not None:
                rows, cols = self._crop5

            X_ret = np.empty((bs, channels, rows, cols), dtype=X_batch.dtype)
            Y_ret = np.empty((bs, Y_batch.shape[1]), dtype=Y_batch.dtype)
            S = np.empty((bs,), dtype=np.int32)

            idx = 0

            for s in range(samples):
                for n in range(self._num):
                    sample = np.rollaxis(X_batch[s, :, :, :], 0, 3)

                    for proc in self._processors:
                        sample = proc.process(sample)

                    if self._crop5 is None:
                        X_ret[idx, :, :, :] = np.rollaxis(sample, 2)
                        Y_ret[idx, :] = Y_batch[s, :]
                        S[idx] = s

                        idx += 1
                    else:
                        for p in c5p:
                            samplec = p.process(sample)
                            X_ret[idx, :, :, :] = np.rollaxis(samplec, 2)
                            Y_ret[idx, :] = Y_batch[s, :]
                            S[idx] = s

                            idx += 1

            yield X_ret, Y_batch, S
