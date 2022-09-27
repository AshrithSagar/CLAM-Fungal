import os
import numpy as np
import cv2 as cv
import argparse

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix


parser = argparse.ArgumentParser(description='Image patches to numpy')
parser.add_argument('--source', type = str,
                    help='Path to folder containing the image folders of patches')


if __name__ == '__main__':
    args = parser.parse_args()

    patch_dir = args.source

    # Loading ResNet50 wit imagenet weights, include_top means that we loading model without last fully connected layers
    model = ResNet50(weights = 'imagenet', include_top = False)

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
            print(features)

            break
        break
