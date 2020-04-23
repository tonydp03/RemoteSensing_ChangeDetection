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
import cdModels
from osgeo import gdal
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--size', type=int, default=128)
parser.add_argument('-cpt', type=int, default=500) # Number of crops per tiff
parser.add_argument('--channels', '-ch', type=int, default=13) # Number of channels
parser.add_argument('--loss', '-l', type=str, default='bce', help='bce, bced or dice')
parser.add_argument('--model', type=str, default='EF', help='EF, Siam or SiamDiff')
parser.add_argument('--city', type=str, default='all', help='Type "all" to perform inference on the full test dataset')

args = parser.parse_args()

img_size = args.size
cpt = args.cpt
channels = args.channels
mod = args.model
classes = 1
dataset_dir = '../CD_wOneraDataset/OneraDataset_Images/'
model_dir = 'models/' + mod + '/'
infres_dir = 'results/'
f = args.city
loss = args.loss

model_name = mod+'_'+str(img_size)+'_cpt-'+str(cpt)+'-'+loss+'_'+str(channels)+'channels'
history_name = model_name + '_history'

model_dir = model_dir + model_name + '/' #'_old/'
hist_dir = model_dir + 'histories/'

model_name = mod+'_'+str(img_size)+'_cpt-'+str(cpt)+'-'+loss+'_'+str(channels)+'channels-final'

os.makedirs(infres_dir, exist_ok=True)

if(f=='all'):
    ftest = open(dataset_dir + 'test.txt', 'r')
    folders = ftest.read().split(',')
    ftest.close()
else:
    folders=[f]

# Load the model
if(loss=='bced'):
    model = K.models.load_model(model_dir + model_name + '.h5', custom_objects={'weighted_bce_dice_loss': cdModels.weighted_bce_dice_loss})
else:
    model = K.models.load_model(model_dir + model_name + '.h5')    
model.summary()
print("Model loaded!")

for f in folders:
    print('Running inference on city: %s' %f)
    # Build raster, pad it and crop it to get the input images
    raster1 = cdUtils.build_raster(dataset_dir + f + '/imgs_1_rect/', channels)
    raster2 = cdUtils.build_raster(dataset_dir + f + '/imgs_2_rect/', channels)
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
    cm += 1

    res_dir = infres_dir + f

    os.makedirs(res_dir, exist_ok=True)

    # Now create the georeferenced change map
    for img in os.listdir(dataset_dir + f + '/imgs_1/'):
        if(img.endswith('B04.tif') and img.startswith('S')):
            geoTiff = gdal.Open(dataset_dir + f + '/imgs_1/' + img)


    # print(gdal.Info(geoTiff))
    cdUtils.createGeoCM(res_dir + '/' + model_name + '.tif', geoTiff, cm)

    print('Georeferenced change map created at %s' %res_dir)