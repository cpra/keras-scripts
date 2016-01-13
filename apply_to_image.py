
'''
Apply a trained Keras net to an image in a sliding window fashion.
~ Christopher
'''

from kutils import utils

import numpy as np
import skimage.io
import skimage.color
import matplotlib.pyplot as plt

import os
import argparse
import json

# parse args

parser = argparse.ArgumentParser(description='Apply a trained Keras network that was trained using `train.py` to an image or a collection of images in a sliding window fashion.')
parser.add_argument('netinfo', type=str, help='Path to a `.info` file written by `train.py`')
parser.add_argument('image', type=str, help='Path to the image to process, or to a folder that contains images.')
parser.add_argument('patch_offset', type=int, help='Offset between extracted patches in pixels.')
parser.add_argument('--channel_order', type=str, default='RGB', help='Order of channels expected by the net (typically `BGR` if OpenCV was used, otherwise `RGB`)')
args = parser.parse_args()

assert(os.path.isfile(args.netinfo))
assert(os.path.exists(args.image))
assert(args.channel_order in ('RGB', 'BGR'))
assert(args.patch_offset > 0)

# search for images


def is_image_file(path):
    return os.path.isfile(path) and path[-4:].lower() in ('.jpg', '.png', '.bmp', 'tiff')


images = [args.image] if is_image_file(args.image) else [os.path.join(args.image, f) for f in os.listdir(args.image) if is_image_file(os.path.join(args.image, f))]
assert(images)

print('Found {} images'.format(len(images)))

# load data

print('Loading "{}" ...'.format(args.netinfo))

with open(args.netinfo, 'r') as f:
    netinfo = json.load(f)

# load and compile net

model = utils.load_compile_model(netinfo, args.netinfo, True)

patchsz = model.layers[0].input_shape[2:]
print('Patch size: {}'.format(patchsz))

# process images

for imp in images:
    print('Processing {} ...'.format(imp))

    # load and preprocess image

    im = skimage.io.imread(imp)
    if im.ndim > 2 and args.channel_order == 'BGR':
        im = im[:, :, [2, 1, 0]]

    imf = im.astype(np.float32)

    if netinfo['preprocess']['demean']:
        for i, m in enumerate(netinfo['preprocess']['channel_means']):
            imf[:, :, i] -= m

    imf /= netinfo['preprocess']['divide']

    # classify patches

    pextractor = utils.SlidingWindowPatchExtractor(imf, patchsz, args.patch_offset)
    batchgen = utils.PatchExtractorMinibatchWrapper(pextractor, 64)

    pmap = np.zeros((im.shape[0], im.shape[1], netinfo['num_classes']), dtype=np.float32)  # P(px = i)
    cmap = np.zeros(im.shape[:2], dtype=np.int32)  # num of overlapping processed patches for each pixel

    for X, pos in batchgen:
        predictions = model.predict(X, X.shape[0])

        for i in range(X.shape[0]):
            y0 = pos[i, 0]
            x0 = pos[i, 1]

            for c in range(netinfo['num_classes']):
                pmap[y0:y0+patchsz[0], x0:x0+patchsz[1], c] += predictions[i, c]

            cmap[y0:y0+patchsz[0], x0:x0+patchsz[1]] += 1

    for c in range(pmap.shape[2]):
        pmap[:, :, c] = np.divide(pmap[:, :, c], np.maximum(cmap, 1))

    # visualize

    vis = np.zeros((cmap.shape[0], cmap.shape[1], 3), dtype=np.float32)
    for c in range(min(3, netinfo['num_classes'])):
        vis[:, :, c] = pmap[:, :, c]

    pp = plt.imshow(vis)
    plt.show()
