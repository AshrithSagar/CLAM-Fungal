# Loading all necessary libraries and modules
import os
import numpy as np
import cv2 as cv

# import tensorflow.keras
# from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

# from keras.applications import ResNet50
# from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50


from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading ResNet50 wit imagenet weights, include_top means that we loading model without last fully connected layers
model = ResNet50(weights = 'imagenet', include_top = False)

# Read image
orig = cv.imread('../image_sets/patches/F005a02/F005a02.tif')

# Convert image to RGB from BGR (another way is to use "image = image[:, :, ::-1]" code)
orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)

# Resize image to 224x224 size
image = cv.resize(orig, (224, 224)).reshape(-1, 224, 224, 3)

# We need to preprocess imageto fulfill ResNet50 requirements
image = preprocess_input(image)

# Extracting our features
features = model.predict(image)

print(features.shape)