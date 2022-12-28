import os
import yaml
import argparse

# import h5py
import numpy as np
from PIL import Image

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features using RESNET | Keras Implementation')
    parser.add_argument('-c', '--config', type = str,
                        help='Path to the config file')

    parser.add_argument('--patch_dir', type = str,
        help='Path to folder containing the image folders of patches')
    parser.add_argument('--feat_dir', type = str,
        help='Path to folder for storing the feature vectors')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['extract_features_resnet_torch']  # Change not required

    patch_dir = args['patch_dir']
    feat_dir = args['feat_dir']


# ----------------------------------------------------------------
# Create feat_dir if not exists.
# Not properly fixed
if not os.path.exists(feat_dir):
    try:
        print("Features directory doesn't exist. Creating ...")
        os.mkdir(feat_dir, exist_ok=True)
    except:
        print("ERROR: Cannot create the Features directory")

# Loading ResNet50 wit imagenet weights, include_top means that we loading model without last fully connected layers
model = ResNet50(weights = 'imagenet', include_top = False)

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

    features = []
    patch_folder = os.path.join(patch_dir, folder)
    for patch_file in sorted(os.listdir(patch_folder)):
        img_path = os.path.join(patch_folder, patch_file)

        # Get coord in [x, y] format
        coord = img_path.split("/")
        coord = coord[-1]
        coord = coord.split(".")[-2]
        coord = coord.split("_")
        coord = [int(coord[-2])/256, int(coord[-1])/256]

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

        # Group the features
        features.append(feature)

        break
    print("="*15)
    break
