import os
import argparse
from PIL import Image
from itertools import product

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patchify images')
    parser.add_argument('--source', type = str,
                        help='Path to folder containing the image files')
    parser.add_argument('--dest', type = str,
                        help='Path to folder for storing the patches')
    parser.add_argument('--patch_size', type = int, default=256,
                        help='patch_size')

    args = parser.parse_args()
	input_dir = args.source
	output_dir = args.dest
	patch_size = args.patch_size

if not input_dir:
    # Path to folder containing the image files
    # input_dir = "/home/keerthanaprasad/RajithaKV/ROI_Detection/F_a/F_a_original/"
    input_dir = "/home/keerthanaprasad/RajithaKV/ROI_Detection/NF_a/"

if not output_dir:
    # Path to folder for storing the patches
    output_dir = "/home/keerthanaprasad/RajithaKV/ROI_Detection/CLAM_model/CLAM_1/image_sets/patches/"

if not patch_size:
    patch_size = 256


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
# main
# --------------------------------

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

for filename in os.listdir(input_dir):
    name, ext = os.path.splitext(filename)
    output_patches_dir = os.path.join(output_dir, name)

    if not os.path.isdir(output_patches_dir):
        os.mkdir(output_patches_dir)

    tile(filename, input_dir, output_patches_dir, patch_size)
