
'''
Visualize how the learning rate progresses over time.
~ Christopher
'''

import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser(description='Visualize the training rate change over time.')
parser.add_argument('--samples', type=int, help='Number of samples in the training set')
parser.add_argument('--batchsize', type=int, help='Minibatch size during training')
parser.add_argument('--lr', type=float, nargs='+', help='Initial learnign rate (multiple values supported)')
parser.add_argument('--decay', type=float, nargs='+', help='Decay (multiple values supported)')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to visualize')
args = parser.parse_args()

# process

print('Iterations per epoch: {}'.format(args.samples / args.batchsize))
print('Total number of iterations after {} epochs: {}'.format(args.epochs, args.epochs * args.samples / args.batchsize))

it = args.samples // args.batchsize

legends = []

for l in args.lr:
    for d in args.decay:
        epochs = []
        lrs = []

        for e in range(args.epochs):
            rate = l * (1.0 / (1.0 + d * e*it))

            epochs.append(e)
            lrs.append(rate)

        legends.append('{} - {}'.format(l, d))
        plt.plot(epochs, lrs)

# show

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend(legends)
plt.gca().set_yscale('log')
plt.show()
