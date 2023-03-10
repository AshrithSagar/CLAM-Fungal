import os
import numpy as np
import cv2 as cv
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import imgaug.augmenters as iaa
import h5py

from modules.resnet_custom import resnet50_baseline
from modules.utils import print_network, collate_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features using RESNET')
    parser.add_argument('-c', '--config', type = str,
                        help='Path to the config file')

    parser.add_argument('--patch_dir', type = str,
        help='Path to folder containing the image folders of patches')
    parser.add_argument('--feat_dir', type = str,
        help='Path to folder for storing the feature vectors')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['extract_features_resnet_torch']

    patch_dir = args['patch_dir']
    feat_dir = args['feat_dir']


# ----------------------------------------------------------------
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
    filename = str(folder).split("/")[-1]
    filePath = os.path.join(feat_dir, filename+'.pt')
    # Run only if file doesn't already exist
    if os.path.exists(filePath):
        print("Skipping File:", filename)
        continue
    print("Running on File:", filename)

    patch_folder = os.path.join(patch_dir, folder)
    if str(patch_folder).split("/")[-1] == "fungal_vs_nonfungal_resnet_features":
        continue

    dataset = []
    for patch_file in sorted(os.listdir(patch_folder)):
        if not patch_file.endswith(".tif"):
            continue

        img_path = os.path.join(patch_folder, patch_file)

        img = Image.open(img_path)

        img_arr = np.asarray(img)
        # img_arr = np.expand_dims(img_arr, 0)
        # img_PIL = Image.fromarray(img_arr)

        imgs_0 = img_arr  # Original patch
        imgs_1 = iaa.Fliplr(p=1.0).augment_image(img_arr)
        imgs_2 = iaa.Flipud(p=1.0).augment_image(img_arr)
        imgs_3 = iaa.Rotate((90, 90)).augment_image(img_arr)
        imgs_4 = iaa.Rotate((180, 180)).augment_image(img_arr)
        imgs_5 = iaa.Rotate((270, 270)).augment_image(img_arr)

        # Create the dataset loader
        imgs = [torch.tensor(imgs_0), torch.tensor(imgs_1), torch.tensor(imgs_2),
                torch.tensor(imgs_3), torch.tensor(imgs_4), torch.tensor(imgs_5)]

        # Get coord in [x, y] format
        coord = img_path.split("/")
        coord = coord[-1]
        coord = coord.split(".")[-2]
        coord = coord.split("_")
        coord = [int(coord[-2])/256, int(coord[-1])/256]

        dataset.append([imgs, coord])

    loader = DataLoader(dataset=dataset, batch_size=1)
    all_features = []
    for count, data in enumerate(loader):
        with torch.no_grad():
            coord = data[1]
            batches = data[0]
            batch_features = []
            for batch in batches:
                batch = torch.unsqueeze(batch, 0)
                batch = batch.reshape([1, 3, 256, 256])
                batch = batch.to(device, non_blocking=True)
                batch = batch.float()

                feature = model(batch)
                feature = feature.cpu().numpy()
                feature = torch.from_numpy(feature)
                feature = np.expand_dims(feature, 0)

                # Group the features, for augmentations of a single patch
                batch_features.append(feature)

            # Group the features, for all patches
            all_features.append(batch_features)

    # To Tensor
    all_features = np.asarray(all_features, dtype="float32")
    all_features = torch.tensor(all_features)

#     print(all_features, " || ", filePath)
#     print("Features size: ", all_features.shape)
    torch.save(all_features, filePath)

    # Save the .hdf5
    # hf = h5py.File('data.h5', 'w')

    print("="*15)
