# To convert image patches to numpy array

import os
import argparse
import h5py
import numpy as np
import PIL


if __name__ == '__main__':
    args = parser.parse_args()

    patch_dir = args.source
    store_dir = args.dest

    for name in os.listdir(patch_dir):
    	img_path = os.path.join(patch_dir, name)
        img = PIL.Image.open(img_path)
		
		img_arr = np.asarray(img)
		img_PIL = PIL.Image.fromarray(img_arr)

		print(img_PIL)

		print("="*50)
        
        img.close()
