# coding: utf-8

"""
Change Detection models Sentinel-2 datasets

@Author: Tony Di Pilato

Created on Wed Feb 19, 2020
"""

import tensorflow as tf
from tensorflow import keras as K
import numpy as np

###### WEIGHTED BCE + DICE LOSS ######

def dice_coef(y_true, y_pred, smooth=1):
    y_true = y_true[:, :, :, -1]
    y_pred = y_pred[:, :, :, -1]
    intersection = K.backend.sum(y_true * y_pred)
    union = K.backend.sum(y_true) + K.backend.sum(y_pred)
    return ((2. * intersection + smooth) / (union + smooth))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# def weighted_bce_dice_loss(y_true,y_pred):
#     class_loglosses = K.backend.mean(K.backend.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2]) # mean bce over all the pixels of an image
#     # weights = [0.5, 23]
#     weights = [0.1, 0.9]
#     weighted_bce = K.backend.sum(class_loglosses * K.backend.constant(weights))
#     return weighted_bce + dice_coef_loss(y_true, y_pred)

# NEW VERSION ---- Should be more correct
def weighted_bce_dice_loss(y_true,y_pred):
    bce = K.backend.binary_crossentropy(y_true, y_pred)
    # Apply the weights
    # weights = [0.5, 23]
    # weights = [0.1, 0.9]
    weights = [0.02, 0.98]
    weight_vector = y_true * weights[0] + (1. - y_true) * weights[1] # the higher weight multiplies the term of the crossentropy corresponding to the class with less samples
    wbce = weight_vector * bce
    weighted_bce = K.backend.mean(wbce)
    return weighted_bce + dice_coef_loss(y_true, y_pred)


##### EARLY FUSION MODEL #####

def EF_UNet_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3, mode='None', axis=3):   
    x = K.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(input_tensor)
    x0 = x
    x = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=axis)(x)
    x = K.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(x)
    if mode == 'residual':
        x = K.layers.Add(name='resi' + stage)([x, x0])
    x = K.layers.BatchNormalization(name='bn' + stage+ '_2', axis=axis)(x)
    x = K.layers.Dropout(0.25, name='dropout'+ stage)(x)
    return x

def EF_UNet(input_shape, classes=1, loss='bce'):
    mode = 'None'
    loss_dict = {'bce':'binary_crossentropy', 'wbced': weighted_bce_dice_loss, 'dice':dice_coef_loss}
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    # Left side of the U-Net
    input_1 = K.Input(shape=input_shape, name='input_1')
    input_2 = K.Input(shape=input_shape, name='input_2')
    inputs = K.layers.Concatenate(axis=bn_axis)([input_1,input_2])

    conv1 = EF_UNet_ConvUnit(inputs, stage='1', nb_filter=nb_filter[0], mode=mode, axis=bn_axis)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = EF_UNet_ConvUnit(pool1, stage='2', nb_filter=nb_filter[1], mode=mode, axis=bn_axis)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = EF_UNet_ConvUnit(pool2, stage='3', nb_filter=nb_filter[2], mode=mode, axis=bn_axis)
    pool3 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = EF_UNet_ConvUnit(pool3, stage='4', nb_filter=nb_filter[3], mode=mode, axis=bn_axis)
    pool4 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottom of the U-Net
    conv5 = EF_UNet_ConvUnit(pool4, stage='5', nb_filter=nb_filter[4], mode=mode, axis=bn_axis)
    
    # Right side of the U-Net
    up1 = K.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(conv5)
    merge1 = K.layers.concatenate([conv4,up1], axis=bn_axis)
    conv6 = EF_UNet_ConvUnit(merge1, stage='6', nb_filter=nb_filter[3], mode=mode, axis=bn_axis)

    up2 = K.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(conv6)
    merge2 = K.layers.concatenate([conv3,up2], axis=bn_axis)
    conv7 = EF_UNet_ConvUnit(merge2, stage='7', nb_filter=nb_filter[2], mode=mode, axis=bn_axis)

    up3 = K.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(conv7)
    merge3 = K.layers.concatenate([conv2,up3], axis=bn_axis)
    conv8 = EF_UNet_ConvUnit(merge3, stage='8', nb_filter=nb_filter[1], mode=mode, axis=bn_axis)
    
    up4 = K.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(conv8)
    merge4 = K.layers.concatenate([conv1,up4], axis=bn_axis)
    conv9 = EF_UNet_ConvUnit(merge4, stage='9', nb_filter=nb_filter[0], mode=mode, axis=bn_axis)

    # Output layer of the U-Net with a softmax activation
    output = K.layers.Conv2D(classes, (1, 1), activation='sigmoid', name='output', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(conv9)

    model = K.Model(inputs=[input_1,input_2], outputs=output, name='EarlyFusion-UNET')

    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4), loss = loss_dict[loss])    
    
    return model


##### SIAMESE model

def Siam_UNet(input_shape,classes=1, loss='bce'):
    nb_filter = [32, 64, 128, 256, 512]
    loss_dict = {'bce':'binary_crossentropy', 'wbced': weighted_bce_dice_loss, 'dice':dice_coef_loss}

    input1 = K.Input(shape=input_shape, name='input_1')
    input2 = K.Input(shape=input_shape, name='input_2')

    # Stage 1 - Start of the left side
    stage = '1'
    conv11 = K.layers.Conv2D(nb_filter[0], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn11 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv12 = K.layers.Conv2D(nb_filter[0], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn12 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop1 = K.layers.Dropout(0.25, name='drop'+ stage)

    x11 = drop1(bn12(conv12(bn11(conv11(input1)))))
    x11_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_1')(x11)
    
    x12 = drop1(bn12(conv12(bn11(conv11(input2)))))
    x12_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_2')(x12)
        
    # Stage 2
    stage = '2'
    conv21 = K.layers.Conv2D(nb_filter[1], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn21 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv22 = K.layers.Conv2D(nb_filter[1], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn22 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop2 = K.layers.Dropout(0.25, name='drop'+ stage)

    x21 = drop2(bn22(conv22(bn21(conv21(x11_p)))))
    x21_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_1')(x21)
 
    x22 = drop2(bn22(conv22(bn21(conv21(x12_p)))))
    x22_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_2')(x22)

    # Stage 3
    stage = '3'
    conv31 = K.layers.Conv2D(nb_filter[2], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn31 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv32 = K.layers.Conv2D(nb_filter[2], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn32 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop3 = K.layers.Dropout(0.25, name='drop'+ stage)

    x31 = drop3(bn32(conv32(bn31(conv31(x21_p)))))
    x31_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_1')(x31)

    x32 = drop3(bn32(conv32(bn31(conv31(x22_p)))))
    x32_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_2')(x32)
    
    # Stage 4
    stage = '4'
    conv41 = K.layers.Conv2D(nb_filter[3], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn41 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv42 = K.layers.Conv2D(nb_filter[3], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn42 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop4 = K.layers.Dropout(0.25, name='drop'+ stage)

    x41 = drop4(bn42(conv42(bn41(conv41(x31_p)))))
    x41_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_1')(x41)

    x42 = drop4(bn42(conv42(bn41(conv41(x32_p)))))

    # UNet bottom - Stage 5
    stage = '5'
    conv51 = K.layers.Conv2D(nb_filter[4], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn51 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv52 = K.layers.Conv2D(nb_filter[4], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn52 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop5 = K.layers.Dropout(0.25, name='drop'+ stage)

    x51 = drop5(bn52(conv52(bn51(conv51(x41_p)))))

    # Stage 6 - Starts of the right side
    stage = '6'
    up1 = K.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(x51) #(merge0)
    merge1 = K.layers.concatenate([up1, x41, x42], axis=3)
    
    conv61 = K.layers.Conv2D(nb_filter[3], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn61 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv62 = K.layers.Conv2D(nb_filter[3], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn62 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop6 = K.layers.Dropout(0.25, name='drop'+ stage)

    x61 = drop6(bn62(conv62(bn61(conv61(merge1)))))
    
    # Stage 7
    stage = '7'
    up2 = K.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(x61)
    merge2 = K.layers.concatenate([up2, x31, x32], axis=3)
    
    conv71 = K.layers.Conv2D(nb_filter[2], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn71 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv72 = K.layers.Conv2D(nb_filter[2], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn72 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop7 = K.layers.Dropout(0.25, name='drop'+ stage)

    x71 = drop7(bn72(conv72(bn71(conv71(merge2)))))
    
    # Stage 8
    stage = '8'
    up3 = K.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(x71)
    merge3 = K.layers.concatenate([up3, x21, x22], axis=3)
    
    conv81 = K.layers.Conv2D(nb_filter[1], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn81 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv82 = K.layers.Conv2D(nb_filter[1], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn82 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop8 = K.layers.Dropout(0.25, name='drop'+ stage)

    x81 = drop8(bn82(conv82(bn81(conv81(merge3)))))

    # Stage 9
    stage = '9'
    up4 = K.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(x81)
    merge4 = K.layers.concatenate([up4, x11, x12], axis=3)
    
    conv91 = K.layers.Conv2D(nb_filter[0], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn91 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv92 = K.layers.Conv2D(nb_filter[0], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn92 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop9 = K.layers.Dropout(0.25, name='drop'+ stage)

    x91 = drop9(bn92(conv92(bn91(conv91(merge4)))))

    # Output layer of the U-Net with a softmax activation
    output = K.layers.Conv2D(classes, (1, 1), activation='sigmoid', name='output', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(x91)

    model = K.Model(inputs=[input1,input2], outputs=output, name='Siam-UNET')
    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4), loss = loss_dict[loss])    

    return model


##### Diff SIAMESE model

def SiamDiff_UNet(input_shape, classes=1, loss='bce'):
    nb_filter = [32, 64, 128, 256]
    bn_axis = 3
    loss_dict = {'bce':'binary_crossentropy', 'bced': weighted_bce_dice_loss, 'dice':dice_coef_loss}

    input1 = K.Input(shape=input_shape, name='input_1')
    input2 = K.Input(shape=input_shape, name='input_2')

    # Stage 1 - Start of the left side
    stage = '1'
    conv11 = K.layers.Conv2D(nb_filter[0], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn11 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv12 = K.layers.Conv2D(nb_filter[0], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn12 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop1 = K.layers.Dropout(0.25, name='drop'+ stage)

    x11 = drop1(bn12(conv12(bn11(conv11(input1)))))
    x11_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_1')(x11)
    
    x12 = drop1(bn12(conv12(bn11(conv11(input2)))))
    x12_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_2')(x12)
        
    # Stage 2
    stage = '2'
    conv21 = K.layers.Conv2D(nb_filter[1], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn21 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv22 = K.layers.Conv2D(nb_filter[1], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn22 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop2 = K.layers.Dropout(0.25, name='drop'+ stage)

    x21 = drop2(bn22(conv22(bn21(conv21(x11_p)))))
    x21_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_1')(x21)
 
    x22 = drop2(bn22(conv22(bn21(conv21(x12_p)))))
    x22_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_2')(x22)

    # Stage 3
    stage = '3'
    conv31 = K.layers.Conv2D(nb_filter[2], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn31 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv32 = K.layers.Conv2D(nb_filter[2], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn32 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop3 = K.layers.Dropout(0.25, name='drop'+ stage)

    x31 = drop3(bn32(conv32(bn31(conv31(x21_p)))))
    x31_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_1')(x31)

    x32 = drop3(bn32(conv32(bn31(conv31(x22_p)))))
    x32_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_2')(x32)
    
    # Stage 4
    stage = '4'
    conv41 = K.layers.Conv2D(nb_filter[3], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn41 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv42 = K.layers.Conv2D(nb_filter[3], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn42 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop4 = K.layers.Dropout(0.25, name='drop'+ stage)

    x41 = drop4(bn42(conv42(bn41(conv41(x31_p)))))
    x41_p = K.layers.MaxPooling2D(pool_size=(2, 2), name='pool'+stage+'_1')(x41)

    x42 = drop4(bn42(conv42(bn41(conv41(x32_p)))))

    # Stage 5 - Starts of the right side
    stage = '5'
    up1 = K.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(x41_p)
    diff1 = K.layers.subtract([x41, x42])
    merge1 = K.layers.concatenate([up1, tf.keras.backend.abs(diff1)], axis=3)
    
    conv51 = K.layers.Conv2DTranspose(nb_filter[3], (3, 3), activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn51 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=bn_axis)
    conv52 = K.layers.Conv2DTranspose(nb_filter[3], (3, 3), activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn52 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=bn_axis)
    conv53 = K.layers.Conv2DTranspose(nb_filter[2], (3, 3), activation='relu', name='conv' + stage + '_3', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn53 = K.layers.BatchNormalization(name='bn' + stage + '_3', axis=bn_axis)
    drop5 = K.layers.Dropout(0.25, name='drop'+ stage)

    x5 = drop5(bn53(conv53(bn52(conv52(bn51(conv51(merge1))))))) 
    
    # Stage 6
    stage = '6' 
    up2 = K.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(x5)
    diff2 = K.layers.subtract([x31, x32])
    merge2 = K.layers.concatenate([up2, tf.keras.backend.abs(diff2)], axis=3)
    
    conv61 = K.layers.Conv2DTranspose(nb_filter[2], (3, 3), activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn61 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=bn_axis)
    conv62 = K.layers.Conv2DTranspose(nb_filter[2], (3, 3), activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn62 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=bn_axis)
    conv63 = K.layers.Conv2DTranspose(nb_filter[1], (3, 3), activation='relu', name='conv' + stage + '_3', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn63 = K.layers.BatchNormalization(name='bn' + stage + '_3', axis=bn_axis)
    drop6 = K.layers.Dropout(0.25, name='drop'+ stage)

    x6 = drop6(bn63(conv63(bn62(conv62(bn61(conv61(merge2))))))) 
    
    # Stage 7
    stage = '7'
    up3 = K.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(x6)
    diff3 = K.layers.subtract([x21, x22])
    merge3 = K.layers.concatenate([up3, tf.keras.backend.abs(diff3)], axis=3)
    
    conv71 = K.layers.Conv2DTranspose(nb_filter[1], (3, 3), activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn71 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=bn_axis)
    conv72 = K.layers.Conv2DTranspose(nb_filter[0], (3, 3), activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn72 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=bn_axis)
    drop7 = K.layers.Dropout(0.25, name='drop'+ stage)

    x7 = drop7(bn72(conv72(bn71(conv71(merge3))))) 

    # Stage 8
    stage = '8'
    up4 = K.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(x7)
    diff4 = K.layers.subtract([x11, x12])
    merge4 = K.layers.concatenate([up4, tf.keras.backend.abs(diff4)], axis=3)
    
    conv81 = K.layers.Conv2DTranspose(nb_filter[0], (3, 3), activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn81 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=bn_axis)
    drop8 = K.layers.Dropout(0.25, name='drop'+ stage)

    x8 = drop8(bn81(conv81(merge4))) 
    
    # Output layer of the U-Net with a softmax activation
    output = K.layers.Conv2D(classes, (1, 1), activation='sigmoid', name='output', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(x8)

    model = K.Model(inputs=[input1,input2], outputs=output, name='new_EarlyFusion-UNET')
    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4), loss = loss_dict[loss])    
    
    return model


# EF v2 --- two outputs

def EFv2_UNet(input_shape, classes=1, loss='bce'):
    mode = 'None'
    loss_dict = {'bce':'binary_crossentropy', 'wbced': weighted_bce_dice_loss, 'dice':dice_coef_loss}
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    # Left side of the U-Net
    input_1 = K.Input(shape=input_shape, name='input_1')
    input_2 = K.Input(shape=input_shape, name='input_2')
    inputs = K.layers.Concatenate(axis=bn_axis)([input_1,input_2])

    conv1 = EF_UNet_ConvUnit(inputs, stage='1', nb_filter=nb_filter[0], mode=mode, axis=bn_axis)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = EF_UNet_ConvUnit(pool1, stage='2', nb_filter=nb_filter[1], mode=mode, axis=bn_axis)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = EF_UNet_ConvUnit(pool2, stage='3', nb_filter=nb_filter[2], mode=mode, axis=bn_axis)
    pool3 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = EF_UNet_ConvUnit(pool3, stage='4', nb_filter=nb_filter[3], mode=mode, axis=bn_axis)
    pool4 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottom of the U-Net
    conv5 = EF_UNet_ConvUnit(pool4, stage='5', nb_filter=nb_filter[4], mode=mode, axis=bn_axis)
    
    # Right side of the U-Net
    up1 = K.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(conv5)
    merge1 = K.layers.concatenate([conv4,up1], axis=bn_axis)
    conv6 = EF_UNet_ConvUnit(merge1, stage='6', nb_filter=nb_filter[3], mode=mode, axis=bn_axis)

    up2 = K.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(conv6)
    merge2 = K.layers.concatenate([conv3,up2], axis=bn_axis)
    conv7 = EF_UNet_ConvUnit(merge2, stage='7', nb_filter=nb_filter[2], mode=mode, axis=bn_axis)

    up3 = K.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(conv7)
    merge3 = K.layers.concatenate([conv2,up3], axis=bn_axis)
    conv8 = EF_UNet_ConvUnit(merge3, stage='8', nb_filter=nb_filter[1], mode=mode, axis=bn_axis)
    
    up4 = K.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(conv8)
    merge4 = K.layers.concatenate([conv1,up4], axis=bn_axis)
    conv9 = EF_UNet_ConvUnit(merge4, stage='9', nb_filter=nb_filter[0], mode=mode, axis=bn_axis)

    # Output layer of the U-Net with a softmax activation for pixel-wise classification
    output_1 = K.layers.Conv2D(classes, (1, 1), activation='sigmoid', name='output_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(conv9)

    # Regression of the % of change pixels included in the image
    flat = K.layers.Flatten()(conv5)
    bnflat = K.layers.BatchNormalization(name='bnflat')(flat)
    dense1 = K.layers.Dense(512, activation='sigmoid', name='dense1', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(bnflat)
    dense2 = K.layers.Dense(128, activation='sigmoid', name='dense2', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense1)
    dense3 = K.layers.Dense(32, activation='sigmoid', name='dense3', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense2)
    
    output_2 = K.layers.Dense(1, activation='sigmoid', name='output_2', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense3)

    model = K.Model(inputs=[input_1,input_2], outputs=[output_1, output_2], name='EarlyFusion-UNET')

    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4), loss = {'output_1':loss_dict[loss], 'output_2':'mse'}, loss_weights={'output_1': 1, 'output_2': 0.1})    
    
    return model


def EF_CNN(input_shape, loss='mse'):
    nb_filter = [16, 32, 64]
    kernel_size = 3
    loss_dict = {'mse':'mse', 'bce':'binary_crossentropy'}

    # Left side of the U-Net
    input_1 = K.Input(shape=input_shape, name='input_1')
    input_2 = K.Input(shape=input_shape, name='input_2')
    inputs = K.layers.Concatenate(axis=3)([input_1,input_2])

    conv1 = K.layers.Conv2D(nb_filter[0], (kernel_size, kernel_size), activation='relu', name='conv1_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(inputs)
    bn1 = K.layers.BatchNormalization(name='bn1_1', axis=3)(conv1)
    conv1 = K.layers.Conv2D(nb_filter[0], (kernel_size, kernel_size), activation='relu', name='conv1_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(bn1)
    bn1 = K.layers.BatchNormalization(name='bn1_2', axis=3)(conv1)
    drop1 = K.layers.Dropout(0.25, name='drop_1')(bn1)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2))(drop1)

    conv2 = K.layers.Conv2D(nb_filter[1], (kernel_size, kernel_size), activation='relu', name='conv2_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(pool1)
    bn2 = K.layers.BatchNormalization(name='bn2_1', axis=3)(conv2)
    conv2 = K.layers.Conv2D(nb_filter[1], (kernel_size, kernel_size), activation='relu', name='conv2_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(bn2)
    bn2 = K.layers.BatchNormalization(name='bn2_2', axis=3)(conv2)
    drop2 = K.layers.Dropout(0.25, name='drop_2')(bn2)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2))(drop2)
    
    conv3 = K.layers.Conv2D(nb_filter[2], (kernel_size-1, kernel_size-1), activation='relu', name='conv3_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(pool2)
    bn3 = K.layers.BatchNormalization(name='bn3_1', axis=3)(conv3)
    conv3 = K.layers.Conv2D(nb_filter[2], (kernel_size-1, kernel_size-1), activation='relu', name='conv3_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(bn3)
    bn3 = K.layers.BatchNormalization(name='bn3_2', axis=3)(conv3)
    drop3 = K.layers.Dropout(0.25, name='drop_3')(bn3)
    # pool3 = K.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(drop3)
    
    flat = K.layers.Flatten()(drop3)
    # bnflat = K.layers.BatchNormalization(name='bnflat')(flat)
    dense1 = K.layers.Dense(64, activation='relu', name='dense1', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(flat)
    dense2 = K.layers.Dense(8, activation='relu', name='dense2', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense1)
    output = K.layers.Dense(1, activation='sigmoid', name='dense3', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(dense2)

    model = K.Model(inputs=[input_1,input_2], outputs=output, name='EarlyFusion-CNN')

    model.compile(optimizer=K.optimizers.Adam(), loss = loss_dict[loss])
    
    return model

