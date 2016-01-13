
'''
Visualize a log file created (and currently written to) by `train.py`.
~ Christopher
'''

import re
import os
import sys
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

# parse args

parser = argparse.ArgumentParser(description='Visualize a log file created by `train.py`.')
parser.add_argument('log', type=str, help='Log file to parse')
parser.add_argument('--batch_results', action='store_true', help='Show batch-level results')
parser.add_argument('--csv', type=str, help='Save results to the specified csv file and quit')
args = parser.parse_args()

assert(os.path.isfile(args.log))
assert(not args.csv or os.path.exists(args.csv))

with open(args.log, 'r') as f:
    for l, line in enumerate(f):
        line = line.strip()
        if l == 0:
            p = json.loads(line)
        else:
            break

i = 0
fs = 0

ret = re.compile('^t (\d+) (\d+) (\d+\.\d+) (\d+\.\d+)$')
ree = re.compile('^e (\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)$')
rev = re.compile('^v (\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)$')

while True:
    i += 1

    idx_train = []
    loss_train = []
    acc_train = []

    initial_acc = 1.0/p['num_classes']
    initial_loss = -np.log(1.0/p['num_classes'])

    epochs_train = [0]
    loss_train_mean = [initial_loss]
    acc_train_mean = [initial_acc]
    loss_train_std = [0]
    acc_train_std = [0]

    epochs_val = [0]
    loss_val_mean = [initial_loss]
    acc_val_mean = [initial_acc]
    loss_val_std = [0]
    acc_val_std = [0]

    fmod = os.path.getmtime(args.log)
    if fmod > fs:
        with open(args.log, 'r') as f:
            for l, line in enumerate(f):
                line = line.strip()
                if l > 1:
                    if line.startswith('t'):
                        m = ret.match(line)

                        idx = int(m.group(1)) + (float(m.group(2)) + 1) / p['num_batches']
                        idx_train.append(idx)

                        loss_train.append(float(m.group(3)))
                        acc_train.append(float(m.group(4)))

                    elif line.startswith('e'):
                        m = ree.match(line)

                        epochs_train.append(int(m.group(1)) + 1)
                        loss_train_mean.append(float(m.group(2)))
                        loss_train_std.append(float(m.group(3)))
                        acc_train_mean.append(float(m.group(4)))
                        acc_train_std.append(float(m.group(5)))

                    elif line.startswith('v'):
                        m = rev.match(line)

                        epochs_val.append(int(m.group(1)) + 1)
                        loss_val_mean.append(float(m.group(2)))
                        loss_val_std.append(float(m.group(3)))
                        acc_val_mean.append(float(m.group(4)))
                        acc_val_std.append(float(m.group(5)))

        if args.csv:
            with open(args.csv, 'w') as f:
                f.write('epoch,loss_train,acc_train,loss_val,acc_val\n')
                assert(np.sum(np.array(epochs_train) - np.array(epochs_val)) == 0)
                for e in range(len(epochs_train)):
                    f.write('{},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(epochs_train[e], loss_train_mean[e], acc_train_mean[e], loss_val_mean[e], acc_val_mean[e]))

            print('Data written to "{}"'.format(args.csv))
            sys.exit(0)

        plt.figure(1)
        plt.clf()

        if args.batch_results:
            plt.plot(idx_train, loss_train, '-b', alpha=0.4)

        plt.errorbar(epochs_train, loss_train_mean, yerr=loss_train_std, fmt='x-b')
        plt.errorbar(epochs_val, loss_val_mean, yerr=loss_val_std, fmt='x-r')
        plt.xticks(epochs_train)
        plt.legend(('train', 'val') if not args.batch_results else ('train batch', 'train', 'val'))
        plt.show(block=False)
        plt.pause(1)

        plt.figure(2)
        plt.clf()

        if args.batch_results:
            plt.plot(idx_train, acc_train, '-b', alpha=0.4)

        plt.errorbar(epochs_train, acc_train_mean, yerr=acc_train_std, fmt='x-b')
        plt.errorbar(epochs_val, acc_val_mean, yerr=acc_val_std, fmt='x-r')
        plt.xticks(epochs_train)
        plt.legend(('train', 'val') if not args.batch_results else ('train batch', 'train', 'val'), loc='lower right')
        plt.show(block=False)
        plt.pause(1)

        fs = fmod

    plt.pause(8)
