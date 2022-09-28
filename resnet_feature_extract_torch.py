import os
import numpy as np
import cv2 as cv
import argparse
import torch


parser = argparse.ArgumentParser(description='Extract features using RESNET')
parser.add_argument('--source', type = str,
                    help='Path to folder containing the image folders of patches')


if __name__ == '__main__':
    args = parser.parse_args()

    patch_dir = args.source

    print(os.path('image_sets/resnet50-19c8e357.pth'))
    model = torch.load(os.path('image_sets/resnet50-19c8e357.pth'))
    print(model)

    for folder in os.listdir(patch_dir):
        patch_folder = os.path.join(patch_dir, folder)
        for patch_file in os.listdir(patch_folder):
            img_path = os.path.join(patch_folder, patch_file)

            # Read image
            orig = cv.imread(img_path)

            # Convert image to RGB from BGR (another way is to use "image = image[:, :, ::-1]" code)
            orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)

            # Resize image to 224x224 size
            image = cv.resize(orig, (224, 224)).reshape(-1, 224, 224, 3)

            # We need to preprocess imageto fulfill ResNet50 requirements
            image = preprocess_input(image)

            # Extracting our features
            features = model.predict(image)

            print(features.shape)

            break
        break
