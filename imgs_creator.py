# To convert image patches to numpy array

import os
import argparse
import h5py
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description='Image patches to numpy')
parser.add_argument('--source', type = str,
                    help='Path to folder containing the image folders of patches')


if __name__ == '__main__':
    args = parser.parse_args()

    patch_dir = args.source

    for folder in os.listdir(patch_dir):
        patch_folder = os.path.join(patch_dir, folder)
        for patch_file in os.listdir(patch_folder):
            img_path = os.path.join(patch_folder, patch_file)
            img = Image.open(img_path)

            img_arr = np.asarray(img)
            img_PIL = Image.fromarray(img_arr)

            img.close()
