# coding: utf-8

"""
Perform inference with a CD UNet/UNet++ model for for Sentinel-2 datasets

@Author: Tony Di Pilato

Created on Wed Feb 19, 2020
"""

import os
import numpy as np
from tensorflow import keras as K
import cdUtils
from osgeo import gdal
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--stride', type=int, default=64)
parser.add_argument('--augmentation', '-a', type=bool, default=False) # Use data augmentation or not
parser.add_argument('-cpt', type=int, default=300) # Number of crops per tiff
# parser.add_argument('--loss', '-l', type=str, default='bce')

args = parser.parse_args()

img_size = args.size
stride = args.stride
aug = args.augmentation
cpt = args.cpt
classes = 1
dataset_dir = '../CD_Sentinel2/tiff/Rome/'
model_dir = 'models/'
infres_dir = 'results/'
# loss = args.loss
# loss = 'binary_crossentropy'
if(aug==True):
    model_name = 'EF_'+str(img_size)+'_aug-'+str(cpt)
else:
    model_name = 'EF_'+str(img_size)+'-'+str(stride)

tiffData = gdal.Open(dataset_dir + 'pre_ro_2018_12_27.tif')
raster1 = tiffData.ReadAsArray()
raster1 = np.moveaxis(raster1, 0, 2) # gdal reads as array with channel-first data format
raster2 = gdal.Open('../CD_Sentinel2/tiff/Rome/post_ro_2019_09_13.tif').ReadAsArray()
raster2 = np.moveaxis(raster2, 0, 2)
raster = np.concatenate((raster1,raster2), axis=2)
padded_raster = cdUtils.pad(raster, img_size)
test_image = cdUtils.crop(padded_raster, img_size, img_size)

# Create inputs for the Neural Network
inputs = np.asarray(test_image, dtype='float32')

# Load model
model = K.models.load_model(model_dir + model_name + '.h5')
model.summary()

print("Model loaded!")

# Perform inference
results = model.predict(inputs)

# Build the complete change map
shape = (padded_raster.shape[0], padded_raster.shape[1], classes)
padded_cm = cdUtils.uncrop(shape, results, img_size, img_size)
cm = cdUtils.unpad(raster.shape, padded_cm)

cm = np.squeeze(cm)
cm = np.rint(cm) # we are only interested in change/unchange
cm = cm.astype(np.uint8)

if not os.path.exists(infres_dir):
    os.mkdir(infres_dir)

# Now create the georeferenced change map
cdUtils.createGeoCM(infres_dir + 'Rome-' + model_name +'.tif', tiffData, cm)

print('Georeferenced change map created at %s' %infres_dir)