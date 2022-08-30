# To glue between create_patches.py and extract_features.py

import os
import argparse
import h5py
import numpy as np


parser = argparse.ArgumentParser(description='Initialize .h5 files')
parser.add_argument('--source', type = str,
                    help='Path to folder containing the image folders of patches')
parser.add_argument('--dest', type = str,
                    help='Path to folder to store the .h5 files')


if __name__ == '__main__':
    args = parser.parse_args()

    patch_dir = args.source
    store_dir = args.dest

    for folder in os.listdir(patch_dir):
        patch_folder = os.path.join(patch_dir, folder)
        for patch_file in os.listdir(patch_folder):
            name, ext = os.path.splitext(patch_file)

            file_path = os.path.join(store_dir, name)+'.h5'
            file = h5py.File(file_path, "w")
            
            file.close()
