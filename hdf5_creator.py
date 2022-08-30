# To glue between create_patches.py and extract_features.py

import os
import argparse

parser = argparse.ArgumentParser(description='Initialize .h5 files')
parser.add_argument('--source', type = str,
                    help='Path to folder containing the image folders of patches')

if __name__ == '__main__':
    args = parser.parse_args()

    for folder in os.listdir(args.source):
        for patch in os.listdir(folder):
            name = patch
            file_path = os.path.join(save_path, name)+'.h5'
            file = h5py.File(file_path, "w")

            file.close()
