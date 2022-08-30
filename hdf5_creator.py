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
            # x, y, cont_idx, patch_level, downsample, downsampled_level_dim, level_dim, img_patch, name, save_path = tuple(first_patch.values())

            name, ext = os.path.splitext(patch_file)

            file_path = os.path.join(store_dir, name)+'.h5'
            file = h5py.File(file_path, "w")
            
            img_patch = np.array(img_patch)[np.newaxis,...]
            dtype = img_patch.dtype

            # Initialize a resizable dataset to hold the output
            img_shape = img_patch.shape
            maxshape = (None,) + img_shape[1:] #maximum dimensions up to which dataset maybe resized (None means unlimited)
            dset = file.create_dataset('imgs', 
                                        shape=img_shape, maxshape=maxshape,  chunks=img_shape, dtype=dtype)

            dset[:] = img_patch
            # dset.attrs['patch_level'] = patch_level
            # dset.attrs['wsi_name'] = name
            # dset.attrs['downsample'] = downsample
            # dset.attrs['level_dim'] = level_dim
            # dset.attrs['downsampled_level_dim'] = downsampled_level_dim

            # if save_coord:
            #     coord_dset = file.create_dataset('coords', shape=(1, 2), maxshape=(None, 2), chunks=(1, 2), dtype=np.int32)
            #     coord_dset[:] = (x,y)

            file.close()
