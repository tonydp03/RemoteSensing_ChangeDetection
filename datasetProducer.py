# coding: utf-8

"""
Create a dataset in .h5 format from Onera Satellite Change Detection dataset

@Author: Tony Di Pilato

Created on Wed Apr 08, 2020
"""

import os
import cdUtils
import numpy as np
from osgeo import osr
from osgeo import gdal
import random
import time
import pandas as pd
import argparse
from sklearn.utils import shuffle 


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('-cpt', type=int, default=500) # Number of crops per tiff
args = parser.parse_args()

img_size = args.size
cpt = args.cpt
channels = 13

dataset_dir = '../CD_wOneraDataset/OneraDataset_Images/'
labels_dir = '../CD_wOneraDataset/OneraDataset_TrainLabels/'
save_dir = 'datasets/'
dataset_name = 'onera_'+str(img_size)+'_cpt-'+str(cpt)


def make_dataset(cpt, crop_size):
    train_images = []
    train_labels = []
    
    # Get the list of folders to open to get rasters
    f = open(dataset_dir + 'train.txt', 'r')
    folders = f.read().split(',')
    f.close()

    for f in folders:
        raster1 = cdUtils.build_raster(dataset_dir + f + '/imgs_1_rect/', channels)
        raster2 = cdUtils.build_raster(dataset_dir + f + '/imgs_2_rect/', channels)
        raster = np.concatenate((raster1,raster2), axis=2)
        cm = gdal.Open(labels_dir + f + '/cm/' + f + '-cm.tif').ReadAsArray()
        cm = np.expand_dims(cm, axis=2)
        cm -= 1 # the change map has values 1 for no change and 2 for change ---> scale back to 0 and 1
        print('*** City %s started ***' %f)
        for i in range(cpt):
            x = random.randint(0,raster.shape[0]-crop_size)
            y = random.randint(0,raster.shape[1]-crop_size)                
            label = cdUtils.trim(cm, x, y, crop_size)
            _, counts = np.unique(label, return_counts=True)
            img = cdUtils.trim(raster, x, y, crop_size)
            if(float(len(counts)==1 or counts[1]/(np.sum(counts)))<0.1):
                n = random.randint(0,5)
                train_images.append(cdUtils.random_transform(img, n))
                train_labels.append(cdUtils.random_transform(label, n))
            else: # if change pixels cover less than 1% of the image, discard the patch
                for n in range(6):
                    train_images.append(cdUtils.random_transform(img, n))
                    train_labels.append(cdUtils.random_transform(label, n))
        print('*** City %s finished ***' %f)

    # Create inputs and labels as arrays
    inputs = np.asarray(train_images, dtype='float32')
    labels = np.asarray(train_labels, dtype='float32')
   
    # Remove doubles
    inputs, indices = np.unique(inputs, axis=0, return_index=True)
    labels = labels[indices]

    # Now shuffle data
    inputs, labels = shuffle(inputs, labels)

    # Flatten arrays and save them as dataframes 
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    flat_labels = labels.reshape(labels.shape[0], -1)

    df_images = pd.DataFrame(data=flat_inputs)
    df_labels = pd.DataFrame(data=flat_labels)

    return df_images, df_labels

start_time = time.time()
df_images, df_labels  = make_dataset(cpt, img_size)
print("CREATING TIME --- %s seconds ---" % (time.time() - start_time))
print('DATASET CREATED')
os.makedirs(save_dir, exist_ok=True)

start_time = time.time()
df_images.to_hdf(save_dir+dataset_name+".h5","images",append=False)
df_labels.to_hdf(save_dir+dataset_name+".h5","labels",append=False)
print("SAVING TIME --- %s seconds ---" % (time.time() - start_time))
print('DATASET SAVED')
