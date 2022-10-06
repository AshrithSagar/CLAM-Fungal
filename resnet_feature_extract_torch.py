import os
import numpy as np
import cv2 as cv
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.resnet_custom import resnet50_baseline


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract(path):
    with torch.no_grad():
        features = model()
        print(features)


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

    for folder in os.listdir(patch_dir):
        patch_folder = os.path.join(patch_dir, folder)
        for patch_file in os.listdir(patch_folder):
            img_path = os.path.join(patch_folder, patch_file)

            features = extract(img_path)

            break
        break
