# coding: utf-8

"""
Plot trained model with Keras API

@Author: Tony Di Pilato

Created on Wed Mar 18, 2020
"""


import os
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--stride', type=int, default=64)
parser.add_argument('--augmentation', '-a', type=bool, default=True) # Use data augmentation or not
parser.add_argument('-cpt', type=int, default=600) # Number of crops per tiff
parser.add_argument('--batch', '-b', type=int, default=32)
parser.add_argument('--channels', '-ch', type=int, default=13) # Number of channels
parser.add_argument('--model', type=str, default='EF', help='EF, new_EF or Siam')
args = parser.parse_args()

batch_size = args.batch
img_size = args.size
channels = args.channels
stride = args.stride
aug = args.augmentation
cpt = args.cpt
mod = args.model
classes = 1
model_dir = 'models/' + mod + '/'
plot_dir = 'plots/' + mod + '/'

if(aug==True):
    test_name = mod+'_'+str(img_size)+'_aug-'+str(cpt)
    model_name = test_name+'_'+str(channels)+'channels'
else:
    test_name = mod+'_'+str(img_size)+'-'+str(stride)
    model_name = test_name+'_'+str(channels)+'channels'

plot_dir = plot_dir+test_name+'/'
os.makedirs(plot_dir, exist_ok=True)


# Load the model
model = tf.keras.models.load_model(model_dir + model_name + '.h5')
model.summary()

tf.keras.utils.plot_model(model, to_file=plot_dir+model_name+'.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)