import os
import argparse
import yaml
from PIL import Image
from itertools import product

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patchify images')
    parser.add_argument('-c', '--config', type = str,
                        help='Path to the config file')
    
    parser.add_argument('--input_dir', type = str,
                        help='Path to folder containing the image files')
    parser.add_argument('--output_dir', type = str,
                        help='Path to folder for storing the patches')
    parser.add_argument('--patch_size', type = int, default=256,
                        help='patch_size')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['patchify']

    input_dir = args.input_dir
    output_dir = args.output_dir
    patch_size = args.patch_size


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


# ----------------------------------------------------------------
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for filename in os.listdir(input_dir):
    name, ext = os.path.splitext(filename)
    output_patches_dir = os.path.join(output_dir, name)

    if not os.path.isdir(output_patches_dir):
        os.mkdir(output_patches_dir)

    tile(filename, input_dir, output_patches_dir, patch_size)
