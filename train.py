
'''
Train a Keras net. Requires a json config file, see `train.example.json` for an example.
~ Christopher
'''

import numpy as np

import os
import json
import argparse
import math

# import args

parser = argparse.ArgumentParser(description='Train a Keras network.')
parser.add_argument('fpath', type=str, help='Path to JSON file of training properties (see `train.example.json`)')
parser.add_argument('--rng', type=int, default=1, help='Random number generator seed to use for reproducible results')
parser.add_argument('--weights', type=str, help='Path to a HDF5 file of model weights to use for initialization')
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train for')
parser.add_argument('--stop_early', type=int, help='If specified, stop learning if the val accuracy did not increase for `n` epochs')
parser.add_argument('--save', type=str, help='File to which to save model weights (excluding file ending, chosen automatically if not specified)')
parser.add_argument('--save_best_only', action='store_true', help='If specified, newer weights overwrite older ones only if the val loss decreased')
parser.add_argument('--log', type=str, default='auto', help='File to which to log to (`auto` means assign automatically, `` means do not log)')
args = parser.parse_args()

assert(os.path.isfile(args.fpath))
assert(not args.weights or os.path.isfile(args.weights))

print('RNG seed: {}'.format(args.rng))
np.random.seed(args.rng)

from kutils import utils
from keras import callbacks as keras_callbacks

# parse properties file

print('Loading properties file "{}"'.format(args.fpath))

with open(args.fpath, 'r') as f:
    props = json.load(f)

print(' Net generator: "{}"'.format(props['netgen']['path']))

# figure out filenames if required

if not args.save:
    args.save = '{}_{}'.format(props['netgen']['path'], os.path.splitext(os.path.basename(props['data']['train'][0]))[0])

if args.log == 'auto':
    args.log = '{}.log'.format(args.save)

print('Saving weights to "{}_epN.h5"'.format(args.save))
if args.log:
    print('Saving training progress log to "{}"'.format(args.log))

assert(not os.path.exists(args.save))
assert(not args.log or not os.path.exists(args.log))

# analyze data

print('Analyzing training data ...')

fpmeans = []
fpnum = []
classes = set()

for fp in props['data']['train']:
    X_train, y_train = utils.load_h5_db(fp, True)

    fpmeans.append(np.mean(X_train, axis=(0, 2, 3)))
    fpnum.append(y_train.size)

    for c in np.unique(y_train):
        classes.add(c)

fpmeans = np.array(fpmeans)
fpnum = np.array(fpnum)
fpweights = fpnum.astype(np.float64) / np.sum(fpnum)

cnmeans = np.average(fpmeans, axis=0, weights=fpweights) if fpnum.size > 1 else fpmeans
cnmeans = cnmeans.ravel()

print('{} training samples'.format(fpnum.sum()))
if props['preprocess']['demean']:
    print('Channel means: {}'.format(cnmeans))

print('{} unique classes: {}'.format(len(classes), classes))

if props['data']['val']:
    num_val = 0
    print('Analyzing validation data ...')
    for fp in props['data']['val']:
        X_val, y_val = utils.load_h5_db(fp, True)
        num_val += X_val.shape[0]
    print('{} validation samples'.format(num_val))

print('Data augmentation: train: "{}", val: "{}"'.format(','.join([p for p in props['augment']['train']]), ','.join([p for p in props['augment']['val']])))

# load net

if args.weights:
    props['weights'] = args.weights

model = utils.load_compile_model_for_training(props, len(classes), args.weights, True)

# init log

log_ = open(args.log, 'w') if args.log else None


def log(s):
    if log_ is not None:
        log_.write('{}\n'.format(s))
        log_.flush()


log(json.dumps({'num_classes': len(classes), 'num_batches': int(math.ceil(fpnum.sum()) / props['data']['batchsize'])}))

logfreq = int((fpnum.sum() / props['data']['batchsize']) / 50)

# setup training

cbs = []
cbs.append(keras_callbacks.BaseLogger())

cbs = keras_callbacks.CallbackList(cbs)
cbs._set_model(model)
cbs._set_params({
    'batch_size': props['data']['batchsize'],
    'nb_epoch': args.epochs,
    'nb_sample': fpnum.sum(),
    'verbose': True,
    'do_validation': True if props['data']['val'] else False,
    'metrics': ['loss', 'acc', 'val_loss', 'val_acc']
})
cbs.on_train_begin()

losses = []
accs = []

monitored_accs = []

e = 0

while True:
    cbs.on_epoch_begin(e)

    # train

    train_losses = []
    train_accs = []

    for fp in props['data']['train']:
        # load current file

        batchgen = utils.BatchGen(X_train, y_train, props['data']['batchsize'], True, len(classes)) if len(props['data']['train']) == 1 else utils.BatchGen.from_file(fp, props['data']['batchsize'], True, len(classes))

        if props['augment']['train']:
            batchgen = utils.MinibatchProcessor(batchgen, props['augment']['train'])

        # train on batches

        b = 0
        for X_batch, Y_batch in batchgen:
            if props['preprocess']['demean']:
                for i, m in enumerate(cnmeans):
                    X_batch[:, i, :, :] -= m

            if props['preprocess']['divide'] != 1:
                X_batch /= props['preprocess']['divide']

            cbs.on_batch_begin(b)
            tloss, tacc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            cbs.on_batch_end(b, {'size': X_batch.shape[0], 'loss': tloss, 'acc': tacc})

            train_losses.append(tloss)
            train_accs.append(tacc)

            if b % logfreq == 0:
                log('t {} {} {:.4f} {:.4f}'.format(e, b, float(tloss), float(tacc)))

            b += 1

    train_losses = np.array(train_losses)
    train_accs = np.array(train_accs)

    log('e {} {:.3f} {:.3f} {:.3f} {:.3f}'.format(e, float(train_losses.mean()), float(train_losses.std()), float(train_accs.mean()), float(train_accs.std())))

    # test

    if props['data']['val']:
        val_losses = []
        val_accs = []

        for fp in props['data']['val']:
            # load current file

            batchgen = utils.BatchGen(X_val, y_val, props['data']['batchsize'], False, len(classes)) if len(props['data']['val']) == 1 else utils.BatchGen.from_file(fp, props['data']['batchsize'], False, len(classes))

            if props['augment']['val']:
                batchgen = utils.MinibatchProcessor(batchgen, props['augment']['val'])

            # train on batches

            for X_batch, Y_batch in batchgen:
                if props['preprocess']['demean']:
                    for i, m in enumerate(cnmeans):
                        X_batch[:, i, :, :] -= m

                if props['preprocess']['divide'] != 1:
                    X_batch /= props['preprocess']['divide']

                vloss, vacc = model.test_on_batch(X_batch, Y_batch, accuracy=True)

                val_losses.append(vloss)
                val_accs.append(vacc)

        val_losses = np.array(val_losses)
        val_accs = np.array(val_accs)

        monitored_accs.append(val_accs.mean())
        cbs.on_epoch_end(e, {'val_loss': val_losses.mean(), 'val_acc': val_accs.mean()})
        log('v {} {:.3f} {:.3f} {:.3f} {:.3f}'.format(e, float(val_losses.mean()), float(val_losses.std()), float(val_accs.mean()), float(val_accs.std())))
    else:
        monitored_accs.append(train_accs.mean())
        cbs.on_epoch_end(e)

    macc = np.array(monitored_accs)

    if not args.save_best_only or np.all(macc[:-1] < macc[-1]):
        wfn = '{}_ep{}.h5'.format(args.save, e+1) if not args.save_best_only else args.save + '.h5'
        ifn = wfn + ".info"

        if not args.save_best_only:
            assert(not os.path.exists(wfn))
            assert(not os.path.exists(ifn))
        else:
            print('Saving model weights')

        model.save_weights(wfn, overwrite=True)

        info = {}
        info['model'] = json.loads(model.to_json())
        info['epoch'] = e+1
        info['weights'] = wfn
        info['train_args'] = vars(args)
        info['num_classes'] = len(classes)
        info['preprocess'] = props['preprocess']
        info['preprocess']['channel_means'] = [float(m) for m in cnmeans]

        with open(ifn, 'w') as f:
            json.dump(info, f, indent=2)

    if args.stop_early and macc.size > args.stop_early and not np.any(macc[-1] > macc[-(args.stop_early+1):-1]):
        print('Accuracy did not improve for {} epochs, stopping'.format(args.stop_early))
        break

    e += 1

    if e == args.epochs:
        print('Target number of epochs reached.')
        print(' Enter `n`, the number of additional epochs, to continue training')
        print(" Enter `n@l` to continue training for `n` epochs at the specified base learning rate (i.e. before decay)")

        inpt = input('Input: ')
        try:
            nd = int(inpt)
            args.epochs += nd
            print('Training for {} more epochs'.format(nd))
        except ValueError:
            if '@' in inpt:
                try:
                    nd = int(inpt.split('@')[0])
                    nlr = float(inpt.split('@')[1])

                    args.epochs += nd
                    model.optimizer.lr.set_value(nlr)

                    print('Training for {} more epochs at rate {}'.format(nd, nlr))
                except ValueError:
                    break
            else:
                break

cbs.on_train_end()
