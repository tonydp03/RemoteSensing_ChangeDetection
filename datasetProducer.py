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
    train_images_pre = []
    train_images_post = []
    train_labels = []
    
    # Get the list of folders to open to get rasters
    f = open(dataset_dir + 'train.txt', 'r')
    folders = f.read().split(',')
    f.close()

    for f in folders:
        raster1 = cdUtils.build_raster(dataset_dir + f + '/imgs_1_rect/', channels)
        raster2 = cdUtils.build_raster(dataset_dir + f + '/imgs_2_rect/', channels)
        cm = gdal.Open(labels_dir + f + '/cm/' + f + '-cm.tif').ReadAsArray()
        cm = np.expand_dims(cm, axis=2)
        cm -= 1 # the change map has values 1 for no change and 2 for change ---> scale back to 0 and 1
        print('*** City %s started ***' %f)
        i = 0
        while(i<cpt):
            x = random.randint(0,cm.shape[0]-crop_size)
            y = random.randint(0,cm.shape[1]-crop_size)                
            label = cdUtils.trim(cm, x, y, crop_size)
            _, counts = np.unique(label, return_counts=True)
            try:
                ratio = float(counts[1]/np.sum(counts))
                if(ratio>=0.01 and ratio < 0.05): # if change pixels cover between 1% and 5% of the patch, apply 1 random transformation
                    img_pre = cdUtils.trim(raster1, x, y, crop_size)
                    img_post = cdUtils.trim(raster2, x, y, crop_size)
                    n = random.randint(0,5)
                    train_images_pre.append(cdUtils.random_transform(img_pre, n))
                    train_images_post.append(cdUtils.random_transform(img_post, n))
                    train_labels.append(cdUtils.random_transform(label, n))
                    i+=1
                elif(ratio>=0.05): # if change pixels cover 5% or more of the patch, save it with all the transformations
                    img_pre = cdUtils.trim(raster1, x, y, crop_size)
                    img_post = cdUtils.trim(raster2, x, y, crop_size)
                    for n in range(6):
                        train_images_pre.append(cdUtils.random_transform(img_pre, n))
                        train_images_post.append(cdUtils.random_transform(img_post, n))
                        train_labels.append(cdUtils.random_transform(label, n))
                    i+=1
                else: # if change pixels cover less than 1% of the image, discard the patch
                    continue
            except: # if the patch is black, it's not possible to calculate ratio as np.unique will be just 0 (all pixels unchanged)
                continue
        print('*** City %s finished ***' %f)

    # Create inputs and labels as arrays
    inputs_pre = np.asarray(train_images_pre)
    inputs_post = np.asarray(train_images_post)
    labels = np.asarray(train_labels)

    # Remove doubles
    inputs_pre = np.unique(inputs_pre, axis=0)
    inputs_post = np.unique(inputs_post, axis=0)
    labels = np.unique(labels, axis=0)

    # Flatten arrays and save them as dataframes 
    flat_inputs_pre = inputs_pre.reshape(inputs_pre.shape[0], -1)
    flat_inputs_post = inputs_post.reshape(inputs_post.shape[0], -1)
    flat_labels = labels.reshape(labels.shape[0], -1)

    df_images_pre = pd.DataFrame(data=flat_inputs_pre)
    df_images_post = pd.DataFrame(data=flat_inputs_post)
    df_labels = pd.DataFrame(data=flat_labels)

    return df_images_pre, df_images_post, df_labels

start_time = time.time()
df_images_pre, df_images_post, df_labels  = make_dataset(cpt, img_size)
print("CREATING TIME --- %s seconds ---" % (time.time() - start_time))
print('DATASET CREATED')

os.makedirs(save_dir, exist_ok=True)

start_time = time.time()
df_images_pre.to_hdf(save_dir+dataset_name+".h5","images_pre",append=False,complevel=1)
df_images_post.to_hdf(save_dir+dataset_name+".h5","images_post",append=False,complevel=1)
df_labels.to_hdf(save_dir+dataset_name+".h5","labels",append=False,complevel=1)
print("SAVING TIME --- %s seconds ---" % (time.time() - start_time))
print('DATASET SAVED')