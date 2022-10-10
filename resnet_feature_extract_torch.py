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


def extract(img_paths):
    dataset = []
    for img_path in img_paths:
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

    for count, data in enumerate(loader):
        with torch.no_grad():
            batch = data[0]
            batch = torch.unsqueeze(batch, 0)
            batch = batch.reshape([1, 3, 256, 256])
            batch = batch.to(device, non_blocking=True)
            batch = batch.float()
            features = model(batch)
            coord = data[1]
            print(count)
            print("Coord", coord)
            print(features)
            print("="*15)


parser = argparse.ArgumentParser(description='Extract features using RESNET')
parser.add_argument('--source', type = str,
    help='Path to folder containing the image folders of patches')
parser.add_argument('--output', type = str,
    help='Path to folder for storing the feature vectors')
args = parser.parse_args()


if __name__ == '__main__':
    patch_dir = args.source
    feat_dir = args.output

    model = resnet50_baseline(pretrained=True)
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.eval()

    img_paths = []
    for folder in os.listdir(patch_dir):
        patch_folder = os.path.join(patch_dir, folder)
        for patch_file in os.listdir(patch_folder):
            img_path = os.path.join(patch_folder, patch_file)
            img_paths.append(img_path)

    features = extract(img_paths)
