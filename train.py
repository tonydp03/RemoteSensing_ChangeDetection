# coding: utf-8

"""
Train a CD UNet/UNet++ model for Sentinel-2 datasets

@Author: Tony Di Pilato

Created on Wed Feb 19, 2020
"""


import os
import numpy as np
#import tensorflow as tf
from tensorflow import keras as K
import cdUtils
from osgeo import gdal
import pandas as pd
import cdModels
import argparse
from sklearn.utils import class_weight


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--stride', type=int, default=64)
parser.add_argument('--epochs', '-e', type=int, default=50)
# parser.add_argument('--loss', '-l', type=str, default='bce')
#parser.add_argument('--model', type=str, default='ef')

args = parser.parse_args()

batch_size = 32
img_size = args.size
channels = 13
stride = args.stride
classes = 1
epochs = args.epochs
dataset_dir = '../CD_wOneraDataset/OneraDataset_Images/'
labels_dir = '../CD_wOneraDataset/OneraDataset_TrainLabels/'
model_dir = 'models/'
hist_dir = 'histories/'
# loss = args.loss
model_name = 'EF-UNet_'+str(img_size)+'-'+str(stride)
history_name = model_name + '_history'

# Get the list of folders to open to get rasters
f = open(dataset_dir + 'train.txt', 'r')
folders = f.read().split(',')
f.close()

# Build rasters, pad them and crop them to get the input images
train_images = []
for f in folders:
    raster1 = cdUtils.build_raster(dataset_dir + f + '/imgs_1_rect/')
    raster2 = cdUtils.build_raster(dataset_dir + f + '/imgs_2_rect/')
    raster = np.concatenate((raster1,raster2), axis=2)
    padded_raster = cdUtils.pad(raster, img_size)
    train_images = train_images + cdUtils.crop(padded_raster, img_size, stride)    

# Read change maps, pad them and crop them to get the ground truths
train_labels = []
for f in folders:
    cm = gdal.Open(labels_dir + f + '/cm/' + f + '-cm.tif').ReadAsArray()
    cm = np.expand_dims(cm, axis=2)
    cm -= 1 # the change map has values 1 for no change and 2 for change ---> scale back to 0 and 1
    padded_cm = cdUtils.pad(cm, img_size)
    train_labels = train_labels + cdUtils.crop(padded_cm, img_size, stride)

# Create inputs and labels for the Neural Network
inputs = np.asarray(train_images, dtype='float16')
labels = np.asarray(train_labels, dtype='float16')

# Compute class weights
flat_labels = np.reshape(labels,[-1])
weights = class_weight.compute_class_weight('balanced', np.unique(flat_labels), flat_labels)
print("**** Weights: ", weights)

# Create the model
model = cdModels.EF_UNet([img_size,img_size,2*channels], classes)
model.summary()

# Train the model
history = model.fit(inputs, labels, batch_size=batch_size, epochs=epochs, class_weight=weights, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

# Save the history for accuracy/loss plotting

if not os.path.exists(hist_dir):
    os.mkdir(hist_dir)

history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + ".h5", "history", append=False)

# Save model and weights

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model.save(model_dir + model_name + ".h5")
print('Trained model saved @ %s ' % model_dir)