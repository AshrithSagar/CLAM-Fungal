import os
import argparse
import yaml
import numpy as np
from PIL import Image
from itertools import product
import cv2

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give Patch level labels')
    parser.add_argument('-c', '--config', type = str,
                        help='Path to the config file')

    parser.add_argument('--source_dir', type = str,
                        help='Path to folder containing the image files')
    parser.add_argument('--patch_dir', type = str,
                        help='Path to folder for storing the patches')
    parser.add_argument('--threshold', type = str,
                        help='Threshold to consider for the annotated images')
    parser.add_argument('--patch_size', type = int, default=256,
                        help='patch_size')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['patch_labels']

    source_dir = args['source_dir']
    patch_dir = args['patch_dir']
    threshold = args['threshold']
    patch_size = args['patch_size']


def tile(filename, dir_in, dir_out, d):
    non_zeros = []
    name, ext = os.path.splitext(filename)
    img = cv2.imread(os.path.join(dir_in, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    w, h = img.size

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        i /= 256
        j /= 256
        out = os.path.join(dir_out, f'{name}_{int(i)}_{int(j)}{ext}')
        im_np = np.asarray(img.crop(box))
        non_zero = np.count_nonzero(im_np)
        
        img.crop(box).save(out)
        non_zeros.append(non_zero)
    print(non_zeros)


# ----------------------------------------------------------------
if not os.path.isdir(patch_dir):
    os.mkdir(patch_dir)

for filename in os.listdir(source_dir):
    name, ext = os.path.splitext(filename)
    output_patches_dir = os.path.join(patch_dir, name)

    if not os.path.isdir(output_patches_dir):
        os.mkdir(output_patches_dir)

    print("Binarizing and Patching", filename)
    tile(filename, source_dir, output_patches_dir, patch_size)
