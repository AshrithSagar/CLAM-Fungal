# -*- coding: utf-8 -*-
"""Copy of Resnet Final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jpuA67NmDbUhWbc4E6IXlpE9srIP0Q0E

**Import all necessary libraries**
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adadelta, RMSprop
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.layers import Input, Add, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pydot
from matplotlib.pyplot import imshow
import scipy.misc
import os 
import glob
import gc
import numpy as np
import pandas as pd
import cv2
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os
import zipfile
from google.colab import drive
import seaborn as sns
from time import time
import datetime
print("tensorflow version:",tf.__version__)

# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from PIL import Image

drive.mount("/content/drive")

"""**Loading Images**

**Train images**
"""

image_directory = '/content/drive/MyDrive/classification/Data/train/'
SIZE = 224
dataset = []  
label=[]

Nonfungal_images = os.listdir(image_directory + 'New non fungal/')
for i, image_name in enumerate(Nonfungal_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'New non fungal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)


fungal_images = os.listdir(image_directory + 'New fungal/')
for i, image_name in enumerate(fungal_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'New fungal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)
print(dataset.shape)
print(label.shape)

"""**Validation images**"""

image_directory = '/content/drive/MyDrive/classification/Data/val/'
SIZE = 224
X_val = []  
y_val=[]

Nonfungal_images = os.listdir(image_directory + 'New non fungal/')
for i, image_name in enumerate(Nonfungal_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'New non fungal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        X_val.append(np.array(image))
        y_val.append(0)


fungal_images = os.listdir(image_directory + 'New fungal/')
for i, image_name in enumerate(fungal_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'New fungal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        X_val.append(np.array(image))
        y_val.append(1)

X_val=np.array(X_val)
y_val=np.array(y_val)
print(X_val.shape)
print(y_val.shape)

"""**Test images**"""

image_directory = '/content/drive/MyDrive/classification/Data/test/'
SIZE = 224
X_test = []   
y_test = []  

Nonfungal_images = os.listdir(image_directory + 'New non fungal/')
for i, image_name in enumerate(Nonfungal_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'New non fungal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        X_test.append(np.array(image))
        y_test.append(0)




fungal_images = os.listdir(image_directory + 'New fungal/')
for i, image_name in enumerate(fungal_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'New fungal/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        X_test.append(np.array(image))
        y_test.append(1)

X_test=np.array(X_test)
y_test=np.array(y_test)
print(X_test.shape)
print(y_test.shape)

# from sklearn.model_selection import train_test_split
# #from tensorflow.keras.utils import to_categorical

# train_img , val_img , train_label , val_label = train_test_split(dataset , label, 
#                                                                  test_size = 0.2,
#                                                                  shuffle = True,
#                                                                  stratify = label)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import tensorflow.keras as keras

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3),pooling='max')

output = resnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)

res_name = []
for layer in resnet.layers:
    res_name.append(layer.name)
resnet.summary()

from tensorflow.keras.utils import plot_model
plot_model(resnet, to_file='/content/drive/MyDrive/Png files/Originalresnet50.png')

"""**Freeze 1/2 of the layers**"""

set_trainable = False
for layer in resnet.layers:
    if layer.name in res_name[-25:]:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

num_classes = 1

model = Sequential()
model.add(resnet)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='/content/drive/MyDrive/Png files/resnet.png')

# model.save('/content/drive/MyDrive/mobile1.h5')

"""**Used Reduce lr**"""

model.compile(loss='binary_crossentropy',
            optimizer = tf.optimizers.Adam(learning_rate=0.00001),
            metrics=['accuracy'])

callbacks_list = [callbacks.ModelCheckpoint(
        filepath = '/content/drive/MyDrive/resnet1-finetune-model.h5',
        verbose = 1,
        monitor = 'val_loss',
        save_best_only = True),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            verbose = 1,
            factor=0.1,
            patience=5,
            mode='min',
            min_lr=1e-8),
        callbacks.CSVLogger(
            filename='/content/drive/MyDrive/resnet1-finetune-model.csv',
            separator = ',',
            append = False),
        callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)]

# Clear any logs from previous runs
#!rm -rf /logs/fit
import time

start = time.time()
# Normal
history = model.fit(dataset,label,  
                  batch_size = 32, 
                  epochs=70,   
                  verbose=1 , 
                  callbacks = callbacks_list,
                  validation_data = (X_val,y_val)
                  )
print("Total time: ", time.time() - start, "seconds") #prints the total training time

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

_, accuracy = model.evaluate(dataset,label)
print("Model accuracy on train data : " + str(accuracy*100))

_, accuracy = model.evaluate(X_val,y_val)
print("Model accuracy on Validation data : " + str(accuracy*100))

_, accuracy = model.evaluate(X_test,y_test)
print("Model accuracy on test data : " + str(accuracy*100))

from sklearn.metrics import classification_report

import seaborn as sns

y_pred = model.predict(X_test)
#y_pred = np.argmax(y_pred, axis=1)
y_pred = np.where(y_pred > 0.5, 1, 0) 
print(classification_report(y_test, y_pred))

test_confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions = y_pred).numpy()
figure1 = plt.figure()
LABELS = ['Non Fungal', 'Fungal']
sns.heatmap(test_confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot = True, cmap=plt.cm.Greens, fmt='d')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.savefig('Test data Confusion Matrix', dpi=250)
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

y_pred = model.predict(X_test)
#y_pred = np.argmax(y_pred, axis=1)

ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, y_pred)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('resnet: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='resnet')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')


# nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test, y_pred)
# auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
# plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Squeezenet (auc = %0.3f)' % auc_keras)
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')

# show the legend
pyplot.legend(loc="lower right")
# show the plot
pyplot.show()



"""**Freezing 1/3rd layers**"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import tensorflow.keras as keras

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3),pooling='max')

output = resnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)

res_name = []
for layer in resnet.layers:
    res_name.append(layer.name)
resnet.summary()

set_trainable = False
for layer in resnet.layers:
    if layer.name in res_name[-33:]:  #first 17 layers freezed 
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

num_classes = 1

model = Sequential()
model.add(resnet)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
            optimizer = tf.optimizers.Adam(learning_rate=0.00001),
            metrics=['accuracy'])

callbacks_list = [callbacks.ModelCheckpoint(
        filepath = '/content/drive/MyDrive/resnet2-finetune-model.h5',  
        verbose = 1,
        monitor = 'val_loss',
        save_best_only = True),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            verbose = 1,
            factor=0.1,
            patience=5,
            mode='min',
            min_lr=1e-8),
        callbacks.CSVLogger(
            filename='/content/drive/MyDrive/resnet2-finetune-model.csv',
            separator = ',',
            append = False),
        callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)]

import time

start = time.time()
# Normal
history = model.fit(dataset,label,  
                  batch_size = 32, 
                  epochs=70,   
                  verbose=1 , 
                  callbacks = callbacks_list,
                  validation_data = (X_val,y_val)
                  )
print("Total time: ", time.time() - start, "seconds")  #prints the total training time

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

_, accuracy = model.evaluate(dataset,label)
print("Model accuracy on train data : " + str(accuracy*100))

_, accuracy = model.evaluate(X_val,y_val)
print("Model accuracy on Validation data : " + str(accuracy*100))

_, accuracy = model.evaluate(X_test,y_test)
print("Model accuracy on test data : " + str(accuracy*100))

from sklearn.metrics import classification_report

import seaborn as sns

y_pred = model.predict(X_test)
#y_pred = np.argmax(y_pred, axis=1)
y_pred = np.where(y_pred > 0.5, 1, 0) 
print(classification_report(y_test, y_pred))

test_confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions = y_pred).numpy()
figure1 = plt.figure()
LABELS = ['Non Fungal', 'Fungal']
sns.heatmap(test_confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot = True, cmap=plt.cm.Greens, fmt='d')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.savefig('Test data Confusion Matrix', dpi=250)
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

y_pred = model.predict(X_test)
#y_pred = np.argmax(y_pred, axis=1)

ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, y_pred)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('resnet: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='resnet')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')


# nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test, y_pred)
# auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
# plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Squeezenet (auc = %0.3f)' % auc_keras)
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')

# show the legend
pyplot.legend(loc="lower right")
# show the plot
pyplot.show()

"""**Freezing 1/4rth layers**"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import tensorflow.keras as keras

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3),pooling='max')

output = resnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)

res_name = []
for layer in resnet.layers:
    res_name.append(layer.name)
resnet.summary()

set_trainable = False
for layer in resnet.layers:
    if layer.name in res_name[-37:]:  #first 13 layers freezed
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

num_classes = 1

model = Sequential()
model.add(resnet)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
            optimizer = tf.optimizers.Adam(learning_rate=0.00001),
            metrics=['accuracy'])

callbacks_list = [callbacks.ModelCheckpoint(
        filepath = '/content/drive/MyDrive/resnet3-finetune-model.h5',  
        verbose = 1,
        monitor = 'val_loss',
        save_best_only = True),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            verbose = 1,
            factor=0.1,
            patience=5,
            mode='min',
            min_lr=1e-8),
        callbacks.CSVLogger(
            filename='/content/drive/MyDrive/resnet3-finetune-model.csv',
            separator = ',',
            append = False),
        callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)]

import time

start = time.time()
# Normal
history = model.fit(dataset,label,  
                  batch_size = 32, 
                  epochs=70,   
                  verbose=1 , 
                  callbacks = callbacks_list,
                  validation_data = (X_val,y_val)
                  )
print("Total time: ", time.time() - start, "seconds")  #prints the total training time

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

_, accuracy = model.evaluate(dataset,label)
print("Model accuracy on train data : " + str(accuracy*100))

_, accuracy = model.evaluate(X_val,y_val)
print("Model accuracy on Validation data : " + str(accuracy*100))

_, accuracy = model.evaluate(X_test,y_test)
print("Model accuracy on test data : " + str(accuracy*100))

from sklearn.metrics import classification_report

import seaborn as sns

y_pred = model.predict(X_test)
#y_pred = np.argmax(y_pred, axis=1)
y_pred = np.where(y_pred > 0.5, 1, 0) 
print(classification_report(y_test, y_pred))

test_confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions = y_pred).numpy()
figure1 = plt.figure()
LABELS = ['Non Fungal', 'Fungal']
sns.heatmap(test_confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot = True, cmap=plt.cm.Greens, fmt='d')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.savefig('Test data Confusion Matrix', dpi=250)
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

y_pred = model.predict(X_test)
#y_pred = np.argmax(y_pred, axis=1)

ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, y_pred)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('resnet: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='resnet')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')


# nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test, y_pred)
# auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
# plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Squeezenet (auc = %0.3f)' % auc_keras)
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')

# show the legend
pyplot.legend(loc="lower right")
# show the plot
pyplot.show()



"""**10 layers freezed**"""

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import tensorflow.keras as keras

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3),pooling='max')

output = resnet.layers[-1].output
output = tf.keras.layers.Flatten()(output)
resnet = Model(resnet.input, output)

res_name = []
for layer in resnet.layers:
    res_name.append(layer.name)
resnet.summary()

set_trainable = False
for layer in resnet.layers:
    if layer.name in res_name[-40:]:  #first 10 layers freezed
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout

num_classes = 1

model = Sequential()
model.add(resnet)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
            optimizer = tf.optimizers.Adam(learning_rate=0.00001),
            metrics=['accuracy'])

callbacks_list = [callbacks.ModelCheckpoint(
        filepath = '/content/drive/MyDrive/resnet10-finetune-model.h5',  
        verbose = 1,
        monitor = 'val_loss',
        save_best_only = True),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            verbose = 1,
            factor=0.1,
            patience=5,
            mode='min',
            min_lr=1e-8),
        callbacks.CSVLogger(
            filename='/content/drive/MyDrive/resnet10-finetune-model.csv',
            separator = ',',
            append = False),
        callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)]

import time

start = time.time()
# Normal
history = model.fit(dataset,label,  
                  batch_size = 32, 
                  epochs=70,   
                  verbose=1 , 
                  callbacks = callbacks_list,
                  validation_data = (X_val,y_val)
                  )
print("Total time: ", time.time() - start, "seconds")  #prints the total training time

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

_, accuracy = model.evaluate(dataset,label)
print("Model accuracy on train data : " + str(accuracy*100))

_, accuracy = model.evaluate(X_val,y_val)
print("Model accuracy on Validation data : " + str(accuracy*100))

_, accuracy = model.evaluate(X_test,y_test)
print("Model accuracy on test data : " + str(accuracy*100))

from sklearn.metrics import classification_report

import seaborn as sns

y_pred = model.predict(X_test)
#y_pred = np.argmax(y_pred, axis=1)
y_pred = np.where(y_pred > 0.5, 1, 0) 
print(classification_report(y_test, y_pred))

test_confusion_matrix = tf.math.confusion_matrix(labels=y_test,predictions = y_pred).numpy()
figure1 = plt.figure()
LABELS = ['Non Fungal', 'Fungal']
sns.heatmap(test_confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot = True, cmap=plt.cm.Greens, fmt='d')
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.savefig('Test data Confusion Matrix', dpi=250)
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

y_pred = model.predict(X_test)
#y_pred = np.argmax(y_pred, axis=1)

ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, y_pred)
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('resnet: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='resnet')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')


# nn_fpr_keras, nn_tpr_keras, nn_thresholds_keras = roc_curve(y_test, y_pred)
# auc_keras = auc(nn_fpr_keras, nn_tpr_keras)
# plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Squeezenet (auc = %0.3f)' % auc_keras)
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')

# show the legend
pyplot.legend(loc="lower right")
# show the plot
pyplot.show()




