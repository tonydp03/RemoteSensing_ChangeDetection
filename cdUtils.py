# coding: utf-8

"""
Main functions to preprocess Sentinel-2 Datasets for Change Detection purpose

@Author: Tony Di Pilato

Created on Wed Feb 19, 2020
"""


import os
import numpy as np
from osgeo import osr
from osgeo import gdal
import random

def build_raster(folder, channels):
    filenames = {3:['B02','B03','B04'], # RGB
        4: ['B02','B03','B04', 'B08'], # 10m resolution
        7: ['B01','B02','B03','B04', 'B08', 'B09', 'B10'], #10m + 60m resolution
        10: ['B02','B03','B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11','B12'], #10m + 20m resolution
        13: ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']} # full raster
    bands = [gdal.Open(folder + f + '.tif').ReadAsArray() for f in filenames[channels]]
    raster = np.stack(bands, axis=2)
    return raster

def build_raster_fromMultispectral(dataset, channels):
    band_num = {3:[2,3,4], # RGB
        4: [2,3,4,8], # 10m resolution
        7: [1,2,3,4,8,10,11], #10m + 60m resolution
        10: [2,3,4,5,6,7,8,9,12,13], #10m + 20m resolution
        13: [1,2,3,4,5,6,7,8,9,10,11,12,13]}

    bands = [dataset.GetRasterBand(b).ReadAsArray() for b in band_num[channels]]
    raster = np.stack(bands, axis = 2)
    return raster
    
def pad(img, crop_size):
    h, w, c = img.shape
    n_h = int(h/crop_size)
    n_w = int(w/crop_size)
    w_toadd = (n_w+1) * crop_size - w
    h_toadd = (n_h+1) * crop_size - h
    img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd), (0,0)], mode='constant')
    return img_pad

def crop(img, crop_size, stride):
    cropped_images = []
    h, w, c = img.shape  
    n_h = int(h/stride)
    n_w = int(w/stride)

    for i in range(n_h):
        for j in range(n_w):
            crop_img = img[(i * stride):((i * stride) + crop_size), (j * stride):((j * stride) + crop_size), :]
            if (crop_img.shape) == (crop_size, crop_size, c):
                cropped_images.append(crop_img)
    return cropped_images

def uncrop(shape, crops, crop_size, stride):
    img = np.zeros(shape)
    h, w, c = shape  
    n_h = int(h/stride)
    n_w = int(w/stride)

    for i in range(n_h):
        for j in range(n_w):
            img[(i * stride):((i * stride) + crop_size), (j * stride):((j * stride) + crop_size), :] = crops[i * n_w + j]
    return img

def unpad(shape, img):
    h, w, c = shape
    return img[:h, :w, :]

def getCoord(geoTiff):
    ulx, pixelWidth, b, uly, d, pixelHeight = geoTiff.GetGeoTransform() # b and d are respectively parameters representing x and y rotation respectively
    lrx = ulx + (geoTiff.RasterXSize * pixelWidth)
    lry = uly + (geoTiff.RasterYSize * pixelHeight)

    return [ulx, uly, lrx, lry]

def createGeoCM(cmName, geoTiff, cmArray):
    ulx, pixelWidth, b, uly, d, pixelHeight = geoTiff.GetGeoTransform() # b and d are respectively parameters representing x and y rotation respectively
    originX = int(ulx)
    originY = int(uly)
    
    if(cmArray.ndim == 3):
        cmArray = np.squeeze(cmArray)

    rows = cmArray.shape[0]
    cols = cmArray.shape[1]

    driver = gdal.GetDriverByName('GTiff')
    GDT_dtype = gdal.GDT_Byte
    band_num = 1
    band_id = 0 # first band in case of multiple bands

    outRaster = driver.Create(cmName, cols, rows, band_num, GDT_dtype)
    outRaster.SetGeoTransform((originX, pixelWidth, b, originY, d, pixelHeight))
    outband = outRaster.GetRasterBand(band_id+1)
    outband.WriteArray(cmArray)

    # Now save the change map

    prj = geoTiff.GetProjection()
    outRasterSRS = osr.SpatialReference(wkt=prj)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def trim(img, x, y, crop_size):
    return(img[x:x+crop_size, y:y+crop_size, :])

def random_transform(img, val):
  return {
    0: lambda img: img,
    1: lambda img: np.rot90(img,1),
    2: lambda img: np.rot90(img,2),
    3: lambda img: np.rot90(img,3),
    4: lambda img: np.flipud(img),
    5: lambda img: np.fliplr(img)
    }[val](img)

# def createDataset_fromOnera(aug, cpt, crop_size, stride, channels, folders, dataset_dir, labels_dir):
#     train_images = []
#     train_labels = []
    
#     if(aug==True): # select random crops and apply transformation
#         for f in folders:
#             raster1 = build_raster(dataset_dir + f + '/imgs_1_rect/', channels)
#             raster2 = build_raster(dataset_dir + f + '/imgs_2_rect/', channels)
#             raster = np.concatenate((raster1,raster2), axis=2)
#             cm = gdal.Open(labels_dir + f + '/cm/' + f + '-cm.tif').ReadAsArray()
#             cm = np.expand_dims(cm, axis=2)
#             cm -= 1 # the change map has values 1 for no change and 2 for change ---> scale back to 0 and 1
#             for i in range(cpt):
#                 x = random.randint(0,raster.shape[0]-crop_size)
#                 y = random.randint(0,raster.shape[1]-crop_size)                
#                 img = trim(raster, x, y, crop_size)
#                 label = trim(cm, x, y, crop_size)
#                 n = random.randint(0,5)          
#                 train_images.append(random_transform(img, n))
#                 train_labels.append(random_transform(label, n))
#     else:
#         for f in folders:
#             raster1 = build_raster(dataset_dir + f + '/imgs_1_rect/')
#             raster2 = build_raster(dataset_dir + f + '/imgs_2_rect/')
#             raster = np.concatenate((raster1,raster2), axis=2)
#             cm = gdal.Open(labels_dir + f + '/cm/' + f + '-cm.tif').ReadAsArray()
#             cm = np.expand_dims(cm, axis=2)
#             cm -= 1 # the change map has values 1 for no change and 2 for change ---> scale back to 0 and 1
#             padded_raster = pad(raster, crop_size)
#             train_images = train_images + crop(padded_raster, crop_size, stride)    
#             padded_cm = pad(cm, crop_size)
#             train_labels = train_labels + crop(padded_cm, crop_size, stride)

#     # Create inputs and labels for the Neural Network
#     inputs = np.asarray(train_images, dtype='float32')
#     labels = np.asarray(train_labels, dtype='float32')
    
#     return inputs, labels

def createDataset_fromOnera(aug, cpt, crop_size, stride, channels, folders, dataset_dir, labels_dir):
    train_images = []
    train_labels = []
    
    if(aug==True): # select random crops and apply transformation
        for f in folders:
            raster1 = build_raster(dataset_dir + f + '/imgs_1_rect/', channels)
            raster2 = build_raster(dataset_dir + f + '/imgs_2_rect/', channels)
            raster = np.concatenate((raster1,raster2), axis=2)
            cm = gdal.Open(labels_dir + f + '/cm/' + f + '-cm.tif').ReadAsArray()
            cm = np.expand_dims(cm, axis=2)
            cm -= 1 # the change map has values 1 for no change and 2 for change ---> scale back to 0 and 1
            print('*** City %s started ***' %f)
            for i in range(cpt):
                x = random.randint(0,raster.shape[0]-crop_size)
                y = random.randint(0,raster.shape[1]-crop_size)                
                label = trim(cm, x, y, crop_size)
                _, counts = np.unique(label, return_counts=True)
                img = trim(raster, x, y, crop_size)
                if(float(len(counts)==1 or counts[1]/(np.sum(counts)))<0.1):
                    n = random.randint(0,5)
                    train_images.append(random_transform(img, n))
                    train_labels.append(random_transform(label, n))
                else: # if change pixels cover less than 1% of the image, discard the patch
                    for n in range(6):
                        train_images.append(random_transform(img, n))
                        train_labels.append(random_transform(label, n))
            print('*** City %s finished ***' %f)
    else:
        for f in folders:
            raster1 = build_raster(dataset_dir + f + '/imgs_1_rect/')
            raster2 = build_raster(dataset_dir + f + '/imgs_2_rect/')
            raster = np.concatenate((raster1,raster2), axis=2)
            cm = gdal.Open(labels_dir + f + '/cm/' + f + '-cm.tif').ReadAsArray()
            cm = np.expand_dims(cm, axis=2)
            cm -= 1 # the change map has values 1 for no change and 2 for change ---> scale back to 0 and 1
            padded_raster = pad(raster, crop_size)
            train_images = train_images + crop(padded_raster, crop_size, stride)    
            padded_cm = pad(cm, crop_size)
            train_labels = train_labels + crop(padded_cm, crop_size, stride)

    # Create inputs and labels for the Neural Network
    inputs = np.asarray(train_images, dtype='float32')
    labels = np.asarray(train_labels, dtype='float32')
    
    # Remove doubles
    inputs, indices = np.unique(inputs, axis=0, return_index=True)
    labels = labels[indices]

    return inputs, labels

def getBandNumbers(channels):
    return {
        3:[2,3,4], # RGB
        4: [2,3,4,8], # 10m resolution
        7: [1,2,3,4,8,10,11], #10m + 60m resolution
        10: [2,3,4,5,6,7,8,9,12,13], #10m + 20m resolution
        13: [1,2,3,4,5,6,7,8,9,10,11,12,13]
        }[channels] # full raster