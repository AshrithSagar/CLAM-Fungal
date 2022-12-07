import os
import numpy as np
import cv2 as cv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import h5py

from models.resnet_custom import resnet50_baseline
from utils.utils import print_network, collate_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features using RESNET')
    parser.add_argument('--source', type = str,
        help='Path to folder containing the image folders of patches')
    parser.add_argument('--output', type = str,
        help='Path to folder for storing the feature vectors')

    args = parser.parse_args()
    patch_dir = args.source
    feat_dir = args.output

if not patch_dir:
    patch_dir = "image_sets/patches/"

if not feat_dir:
    feat_dir = "image_sets/features/"

if not actual_feat_dir:
    actual_feat_dir = "image_sets/patches/fungal_vs_nonfungal_resnet_features/pt_files/"

# ----------------------------------------------------------------
# main
# --------------------------------

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create feat_dir if not exists.
# Not properly fixed
if not os.path.exists(feat_dir):
    try:
        print("Features directory doesn't exist. Creating ...")
        os.mkdir(feat_dir, exist_ok=True)
    except:
        print("ERROR: Cannot create the Features directory")

model = resnet50_baseline(pretrained=True)
model = model.to(device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.eval()

# patch_folders = [os.path.join(patch_dir, folder) for folder in sorted(os.listdir(patch_dir))]
# patches_per_image = len(os.listdir(patch_folders[0]))
# print(patches_per_image)

# Create dataset from the image patches
for folder in sorted(os.listdir(patch_dir)):
    patch_folder = os.path.join(patch_dir, folder)
    dataset = []
    for patch_file in sorted(os.listdir(patch_folder)):
        if patch_file == "pt_files":
            continue

        img_path = os.path.join(patch_folder, patch_file)

        img = Image.open(img_path)

        img_arr = np.asarray(img)
        # img_arr = np.expand_dims(img_arr, 0)
        # img_PIL = Image.fromarray(img_arr)

        # Create the dataset loader
        imgs = torch.tensor(img_arr)

        # Get coord in [x, y] format
        coord = img_path.split("/")
        coord = coord[-1]
        coord = coord.split(".")[-2]
        coord = coord.split("_")
        coord = [int(coord[-2])/256, int(coord[-1])/256]

        dataset.append([imgs, coord])

    loader = DataLoader(dataset=dataset, batch_size=1)
    filename = str(folder).split("/")[-1]
    print("File:", filename)

    features = []
    for count, data in enumerate(loader):
        with torch.no_grad():
            coord = data[1]
            batch = data[0]
            batch = torch.unsqueeze(batch, 0)
            batch = batch.reshape([1, 3, 256, 256])
            batch = batch.to(device, non_blocking=True)
            batch = batch.float()

            feature = model(batch)
            feature = feature.cpu().numpy()
            feature = torch.from_numpy(feature)
            feature = np.expand_dims(feature, 0)

            # Group the features
            features.append(feature)

    # To Tensor
    features = np.asarray(features, dtype="float32")
    features = torch.tensor(features)

    print(filename)
    filePath = os.path.join(feat_dir, filename+'.pt')
    print(count, " || ", coord, " || ", features, " || ", filePath)
    # print("Features size: ", features.shape)
    torch.save(features, filePath)

    # Save the .hdf5
    # hf = h5py.File('data.h5', 'w')

    print("="*15)


# !mkdir -p $actual_feat_dir
# !mv $feat_dir/* $actual_feat_dir

# !rm -rf "${actual_feat_dir}fungal_vs_nonfungal_resnet_features.pt"
