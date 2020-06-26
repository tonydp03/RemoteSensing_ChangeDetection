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
parser.add_argument('-cpt', type=int, default=500) # Number of crops per tiff
parser.add_argument('--channels', '-ch', type=int, default=13)
parser.add_argument('--loss', '-l', type=str, default='bce', help='bce, wbced or dice')
parser.add_argument('--model', type=str, default='EF', help='EF, Siam or SiamDiff')
parser.add_argument('--city', type=str, default='rome', help='rome or sot')
parser.add_argument('--year', '-y', type=int, default='2015', help='from 2015 to 2019, take the pair thisyear-nextyear')
parser.add_argument('--histmatched', '-hm', type=bool, default='False', help='select True to use histmatched post images')

args = parser.parse_args()

img_size = args.size
channels = args.channels
cpt = args.cpt
mod = args.model
loss = args.loss
city = args.city
year = str(args.year)+'-'+str(args.year + 1)
hm = args.histmatched
classes = 1
model_dir = '../models/' + mod + '/'

model_name = mod+'_'+str(img_size)+'_cpt-'+str(cpt)+'-'+loss+'_'+str(channels)+'channels'
model_dir = model_dir + model_name + '/'
model_name = model_name + '-final'

# Load the model
if(loss=='wbced'):
    model = K.models.load_model(model_dir + model_name + '.h5', custom_objects={'weighted_bce_dice_loss': cdModels.weighted_bce_dice_loss})
else:
    model = K.models.load_model(model_dir + model_name + '.h5')    
model.summary()
print("Model loaded!")

if(city=='rome'):
    dataset_dir = '../../CD_Sentinel2/tiff/Rome/'
    infres_dir = '../results/rome/'
    cm_name = 'Rome-' + model_name
else: 
    if(hm):
        dataset_dir = '../Stoke_On_Trent/histmatched/' + year + '/'
        cm_name = 'SoThm-' + model_name + '_' + year
    else:
        dataset_dir = '../Stoke_On_Trent/' + year + '/'
        cm_name = 'SoT-' + model_name + '_' + year
    infres_dir = '../results/SoT/'

os.makedirs(infres_dir, exist_ok=True)

for image in os.listdir(dataset_dir):
    if((image.startswith('2') or image.startswith('p')) and ('pre' in image)):
        img_pre = image
    elif((image.startswith('2') or image.startswith('p')) and ('post' in image)):
        img_post = image

print(img_pre)
print(img_post)


tiffData_1 = gdal.Open(dataset_dir + img_pre)
raster1 = cdUtils.build_raster_fromMultispectral(tiffData_1, channels)
tiffData_2 = gdal.Open(dataset_dir + img_post)
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
cdUtils.createGeoCM(infres_dir + cm_name + '.tif', tiffData_1, cm)

print('Georeferenced change map created at %s' %infres_dir)
