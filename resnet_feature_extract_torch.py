import os
import numpy as np
import cv2 as cv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from models.resnet_custom import resnet50_baseline
from utils.utils import print_network, collate_features


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Extract features using RESNET')
parser.add_argument('--source', type = str,
    help='Path to folder containing the image folders of patches')
parser.add_argument('--output', type = str,
    help='Path to folder for storing the feature vectors')
args = parser.parse_args()


if __name__ == '__main__':
    patch_dir = args.source
    feat_dir = args.output

    # Create feat_dir if not exists
    if not os.path.exists(feat_dir):
        try:
            print("Features directory doesn't exist. Creating ...")
            os.mkdir(feat_dir)
        except:
            print("ERROR: Cannot create the Features directory")

    model = resnet50_baseline(pretrained=True)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()

    # Create dataset from the image patches
    dataset = []
    for folder in os.listdir(patch_dir):
        patch_folder = os.path.join(patch_dir, folder)
        for patch_file in os.listdir(patch_folder):
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

            name = str(patch_file)

            dataset.append([imgs, coord, name])

    loader = DataLoader(dataset=dataset, batch_size=1)

    for count, data in enumerate(loader):
        with torch.no_grad():
            filename = data[2]
            coord = data[1]
            batch = data[0]
            batch = torch.unsqueeze(batch, 0)
            batch = batch.reshape([1, 3, 256, 256])
            batch = batch.to(device, non_blocking=True)
            batch = batch.float()

            features = model(batch)
            features = features.cpu().numpy()
            features = torch.from_numpy(features)

            filePath = os.path.join(feat_dir, filename+'.pt')
            print(count, " || ", coord, " || ", features, " || ", filePath)
            # print("Features size: ", features.shape)

            torch.save(features, filePath)
            print("="*15)
