import os
import numpy as np
import cv2 as cv
import argparse
import torch


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Extract features using RESNET')
parser.add_argument('--source', type = str, 
    help='Path to folder containing the image folders of patches')
args = parser.parse_args()


if __name__ == '__main__':
    patch_dir = args.source

    model_path = os.path.join(os.getcwd(), 'image_sets/resnet50-19c8e357.pth')
    model = torch.load(model_path)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    for folder in os.listdir(patch_dir):
        patch_folder = os.path.join(patch_dir, folder)
        for patch_file in os.listdir(patch_folder):
            img_path = os.path.join(patch_folder, patch_file)
            with torch.no_grad():
                features = model(model)


            break
        break
