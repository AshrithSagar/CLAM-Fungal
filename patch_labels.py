import os
import argparse
import yaml
from PIL import Image

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
    parser.add_argument('--patch_size', type = int, default=256,
                        help='patch_size')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['patch_labels']

    source_dir = args['source_dir']
    patch_dir = args['patch_dir']
    patch_size = args['patch_size']


# ----------------------------------------------------------------
if not os.path.isdir(patch_dir):
    os.mkdir(patch_dir)

for filename in os.listdir(source_dir):
    name, ext = os.path.splitext(filename)
    output_patches_dir = os.path.join(patch_dir, name)

    if not os.path.isdir(output_patches_dir):
        os.mkdir(output_patches_dir)

    print("Patching", filename)
    tile(filename, source_dir, output_patches_dir, patch_size)
