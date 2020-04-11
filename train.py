# coding: utf-8

"""
Train a CD UNet/UNet++ model for Sentinel-2 datasets

@Author: Tony Di Pilato

Created on Wed Feb 19, 2020
"""


import os
import numpy as np
import tensorflow as tf
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
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--batch', '-b', type=int, default=32)
parser.add_argument('--augmentation', '-a', type=bool, default=True) # Use data augmentation or not
parser.add_argument('-cpt', type=int, default=500) # Number of crops per tiff
parser.add_argument('--channels', '-ch', type=int, default=13) # Number of channels
parser.add_argument('--loss', '-l', type=str, default='bce', help='bce, bced or dice')
parser.add_argument('--model', type=str, default='EF', help='EF, Siam or SiamDiff')

args = parser.parse_args()

batch_size = args.batch
img_size = args.size
channels = args.channels
stride = args.stride
classes = 1
epochs = args.epochs
aug = args.augmentation
cpt = args.cpt
mod = args.model
dataset_dir = 'datasets/'
model_dir = 'models/' + mod + '/'
hist_dir = 'histories/' + mod + '/'
loss = args.loss

if(aug==True):
    model_name = mod+'_'+str(img_size)+'_cpt-'+str(cpt)+'-'+loss+'_'+str(channels)+'channels'
else:
    model_name = mod+'_'+str(img_size)+'-'+str(stride)+'-'+loss+'_'+str(channels)+'channels'
history_name = model_name + '_history'

# Get the list of folders to open to get rasters
inputs = pd.read_hdf(dataset_dir+'onera_'+str(img_size)+'_cpt-'+str(cpt)+'.h5', 'images').values.reshape(-1,img_size,img_size,2*channels)
labels = pd.read_hdf(dataset_dir+'onera_'+str(img_size)+'_cpt-'+str(cpt)+'.h5', 'labels').values.reshape(-1,img_size,img_size,1)
inputs = inputs.astype('float32')
labels = labels.astype('float32')


if('Siam' in mod):
    inputs_1 = inputs[:,:,:,:channels]
    inputs_2 = inputs[:,:,:,channels:]
    inputs = [inputs_1, inputs_2]
else:
    channels *=2

# Create the model
model = getattr(cdModels, mod+'_UNet')([img_size,img_size,channels], classes, loss)
model.summary()

# Train the model
if(loss=='bce'):
    # Compute class weights
    flat_labels = np.reshape(labels,[-1])
    weights = class_weight.compute_class_weight('balanced', np.unique(flat_labels), flat_labels)
    history = model.fit(inputs, labels, batch_size=batch_size, epochs=epochs, class_weight=weights, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
else:
    history = model.fit(inputs, labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

# Save the history for accuracy/loss plotting
os.makedirs(hist_dir, exist_ok=True)

history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + ".h5", "history", append=False)

# Save model and weights
os.makedirs(model_dir, exist_ok=True)

model.save(model_dir + model_name + ".h5")
print('Trained model saved @ %s ' % model_dir)
