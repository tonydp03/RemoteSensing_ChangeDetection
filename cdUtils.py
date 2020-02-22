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


def build_raster(folder):
    filenames = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
    bands = [gdal.Open(folder + f + '.tif').ReadAsArray() for f in filenames]
    raster = np.stack(bands, axis=2)
    return raster

def build_rasterRGB(folder):
    filenames = ['B02','B03','B04']
    bands = [gdal.Open(folder + f + '.tif').ReadAsArray() for f in filenames]
    raster = np.stack(bands, axis=2)    
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
