import os
import argparse
import yaml
import cv2
import numpy as np
from PIL import Image
from itertools import product

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patchify images')
    parser.add_argument('-c', '--config', type = str,
                        help='Path to the config file')
    
    parser.add_argument('--source_dir', type = str,
                        help='Path to folder containing the image files')
    parser.add_argument('--patch_dir', type = str,
                        help='Path to folder for storing the patches')
    parser.add_argument('--annotation', type = bool,
                        help='Whether to perform on annotated images or not')
    parser.add_argument('--threshold', type = str,
                        help='Threshold to consider for the annotated images')
    parser.add_argument('--patch_size', type = int, default=256,
                        help='patch_size')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['patch_creator']

    source_dir = args['source_dir']
    patch_dir = args['patch_dir']
    annotation = args['annotation']
    threshold = args['threshold']
    patch_size = args['patch_size']


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        i /= 256
        j /= 256
        out = os.path.join(dir_out, f'{name}_{int(i)}_{int(j)}{ext}')
        img.crop(box).save(out)


def tile_scikit(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = load_sample_image(os.path.join(dir_in, filename))
    print('Image shape: {}'.format(img.shape))

    patches = image.extract_patches_2d(img, (256, 256))
    print('Patches shape: {}'.format(patches.shape))

    print(patches)


def tile_annotations(filename, dir_in, dir_out, d):
    patch_scores = []
    name, ext = os.path.splitext(filename)
    img = cv2.imread(os.path.join(dir_in, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_TOZERO)  # Apply thresholding
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB, for PIL Image
    img = Image.fromarray(img)  # Convert to PIL Image
    w, h = img.size

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        i /= 256
        j /= 256
        out = os.path.join(dir_out, f'{name}_{int(i)}_{int(j)}{ext}')

        img_patch = img.crop(box)

        im_np = np.asarray(img_patch)  # Convert to Numpy array
        patch_non_zero = np.count_nonzero(im_np)
        patch_scores.append(patch_non_zero)

        img_patch.save(out)  # Save patch image

    print(patch_scores)


# ----------------------------------------------------------------
if not os.path.isdir(patch_dir):
    os.mkdir(patch_dir)

for filename in os.listdir(source_dir):
    name, ext = os.path.splitext(filename)
    output_patches_dir = os.path.join(patch_dir, name)

    if not os.path.isdir(output_patches_dir):
        os.mkdir(output_patches_dir)

    if annotation:
        print("Binarizing and Patching", filename)
        tile_annotations(filename, source_dir, output_patches_dir, patch_size)
    else:
        print("Patching", filename)
        tile(filename, source_dir, output_patches_dir, patch_size)
