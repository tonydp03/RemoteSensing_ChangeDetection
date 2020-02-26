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
parser.add_argument('--stride', type=int, default=64) # Stride used for patches when data augmentation is not applied
parser.add_argument('--epochs', '-e', type=int, default=50)
parser.add_argument('--augmentation', '-a', type=bool, default=False) # Implement data augmentation or not
parser.add_argument('-cpt', type=int, default=300) # Number of crops per tiff
#parser.add_argument('--loss', '-l', type=str, default='bce')
#parser.add_argument('--model', type=str, default='ef')

args = parser.parse_args()

batch_size = 32
img_size = args.size
channels = 13
stride = args.stride
classes = 1
epochs = args.epochs
aug = args.augmentation
cpt = args.cpt
dataset_dir = '../CD_wOneraDataset/OneraDataset_Images/'
labels_dir = '../CD_wOneraDataset/OneraDataset_TrainLabels/'
model_dir = 'models/'
hist_dir = 'histories/'
# loss = args.loss
if(aug==True):
    model_name = 'EF_'+str(img_size)+'_aug-'+str(cpt)
else:
    model_name = 'EF_'+str(img_size)+'-'+str(stride)
history_name = model_name + '_history'

# Get the list of folders to open to get rasters
f = open(dataset_dir + 'train.txt', 'r')
folders = f.read().split(',')
f.close()

# Create Dataset from Onera
inputs, labels = cdUtils.createDataset_fromOnera(aug, cpt, img_size, stride, folders, dataset_dir, labels_dir)

# Compute class weights
flat_labels = np.reshape(labels,[-1])
weights = class_weight.compute_class_weight('balanced', np.unique(flat_labels), flat_labels)
print("**** Weights: ", weights)

# Create the model
model = cdModels.EF_UNet([img_size,img_size,2*channels], classes)
model.summary()

# Train the model
history = model.fit(inputs, labels, batch_size=batch_size, epochs=epochs, class_weight=weights, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

# Save the history for accuracy/loss plotting

if not os.path.exists(hist_dir):
    os.mkdir(hist_dir)

history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + ".h5", "history", append=False)

# Save model and weights

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model.save(model_dir + model_name + ".h5")
print('Trained model saved @ %s ' % model_dir)