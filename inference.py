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
import cdModels
from osgeo import gdal
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('-cpt', type=int, default=400) # Number of crops per tiff
parser.add_argument('--channels', '-ch', type=int, default=13)
parser.add_argument('--loss', '-l', type=str, default='bce', help='bce, bced or dice')
parser.add_argument('--model', type=str, default='EF', help='EF, Siam or SiamDiff')

args = parser.parse_args()

img_size = args.size
channels = args.channels
cpt = args.cpt
mod = args.model
loss = args.loss
classes = 1
dataset_dir = '../CD_Sentinel2/tiff/Rome/'
model_dir = 'models/' + mod + '/'
infres_dir = 'results/rome/'

model_name = mod+'_'+str(img_size)+'_cpt-'+str(cpt)+'-'+loss+'_'+str(channels)+'channels'
model_dir = model_dir + model_name + '/'
model_name = model_name + '-final'

os.makedirs(infres_dir, exist_ok=True)

# Load the model
if(loss=='bced'):
    model = K.models.load_model(model_dir + model_name + '.h5', custom_objects={'weighted_bce_dice_loss': cdModels.weighted_bce_dice_loss})
else:
    model = K.models.load_model(model_dir + model_name + '.h5')    
model.summary()
print("Model loaded!")

tiffData_1 = gdal.Open(dataset_dir + 'pre_ro_2018_12_27.tif')
raster1 = cdUtils.build_raster_fromMultispectral(tiffData_1, channels)
tiffData_2 = gdal.Open('../CD_Sentinel2/tiff/Rome/post_ro_2019_09_13.tif')
raster2 = cdUtils.build_raster_fromMultispectral(tiffData_2, channels)
raster = np.concatenate((raster1,raster2), axis=2)
padded_raster = cdUtils.pad(raster, img_size)
test_image = cdUtils.crop(padded_raster, img_size, img_size)

# Create inputs for the Neural Network
inputs = np.asarray(test_image, dtype='float32')
inputs_1 = inputs[:,:,:,:channels]
inputs_2 = inputs[:,:,:,channels:]
inputs = [inputs_1, inputs_2]

# Perform inference
results = model.predict(inputs)

# Build the complete change map
shape = (padded_raster.shape[0], padded_raster.shape[1], classes)
padded_cm = cdUtils.uncrop(shape, results, img_size, img_size)
cm = cdUtils.unpad(raster.shape, padded_cm)

cm = np.squeeze(cm)
cm = np.rint(cm) # we are only interested in change/unchange
cm = cm.astype(np.uint8)

# Now create the georeferenced change map
cdUtils.createGeoCM(infres_dir + 'Rome-' + model_name +'.tif', tiffData_1, cm)

print('Georeferenced change map created at %s' %infres_dir)