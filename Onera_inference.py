# coding: utf-8

"""
Perform inference with a CD UNet/UNet++ model for for Sentinel-2 Onera dataset

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
dataset_dir = '../CD_wOneraDataset/OneraDataset_Images/'
model_dir = 'models/'
infres_dir = 'results/'
# loss = args.loss
# loss = 'binary_crossentropy'
if(aug==True):
    model_name = 'EF_'+str(img_size)+'_aug-'+str(cpt)
else:
    model_name = 'EF_'+str(img_size)+'-'+str(stride)

# Get the list of folders to open to get rasters
# folders = rnc.get_folderList(dataset_dir + 'test.txt')
f = open(dataset_dir + 'train.txt', 'r')
folders = f.read().split(',')
f.close()

# Select a folder, build raster, pad it and crop it to get the input images
# f = random.choice(folders)
f = 'rennes'

raster1 = cdUtils.build_raster(dataset_dir + f + '/imgs_1_rect/')
raster2 = cdUtils.build_raster(dataset_dir + f + '/imgs_2_rect/')
raster = np.concatenate((raster1,raster2), axis=2)
padded_raster = cdUtils.pad(raster, img_size)
test_image = cdUtils.crop(padded_raster, img_size, img_size)

# Create inputs for the Neural Network
inputs = np.asarray(test_image, dtype='float16')

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

res_dir = infres_dir + f

if not os.path.exists(res_dir):
    os.mkdir(res_dir)

# Now create the georeferenced change map
######## This should be fixed (check ERROR 4: no such file or directory)

for img in os.listdir(dataset_dir + f + '/imgs_1/'):
    if(img.endswith('B04.tif') and img.startswith('S')):
        geoTiff = gdal.Open(img)

# print(gdal.Info(geoTiff))
cdUtils.createGeoCM(res_dir + '/' + model_name + '.tif', geoTiff, cm)

print('Georeferenced change map created at %s' %res_dir)