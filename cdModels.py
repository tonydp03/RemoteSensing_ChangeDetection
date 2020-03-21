# coding: utf-8

"""
Change Detection models Sentinel-2 datasets

@Author: Tony Di Pilato

Created on Wed Feb 19, 2020
"""

import tensorflow as tf
from tensorflow import keras as K


##### EARLY FUSION MODEL ##### NAME MODIFIED! 

def EF_UNet_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3, mode='None', axis=3):   
    x = K.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(input_tensor)
    x = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=axis)(x)
    x0 = x
    x = K.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(x)
    x = K.layers.BatchNormalization(name='bn' + stage+ '_2', axis=axis)(x)
    if mode == 'residual':
        x = K.layers.Add(name='resi' + stage)([x, x0])
    x = K.layers.Dropout(0.25, name='dropout'+ stage)(x)
    return x

def EF_UNet(input_shape, classes=1):
    mode = 'residual'
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    # Left side of the U-Net
    inputs = K.Input(shape=input_shape, name='input')

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

    model = K.Model(inputs=inputs, outputs=output, name='EarlyFusion-UNET')

    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4), loss = 'binary_crossentropy')    
    
    return model


##### SIAMESE model

def Siam_UNet(input_shape,classes=1):
    nb_filter = [32, 64, 128, 256, 512]

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
    up1 = K.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(x51)
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

    model = K.Model(inputs=[input1,input2], outputs=output, name='new_EarlyFusion-UNET')
    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4), loss = 'binary_crossentropy')    
    
    return model



##### Diff SIAMESE model

def SiamDiff_UNet(input_shape,classes=1):
    nb_filter = [32, 64, 128, 256, 512]

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
    up1 = K.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(x51)
    diff1 = K.layers.subtract([x41, x42])
    merge1 = K.layers.concatenate([up1, tf.keras.backend.abs(diff1)], axis=3)
    
    conv61 = K.layers.Conv2D(nb_filter[3], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn61 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv62 = K.layers.Conv2D(nb_filter[3], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn62 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop6 = K.layers.Dropout(0.25, name='drop'+ stage)

    x61 = drop6(bn62(conv62(bn61(conv61(merge1)))))
    
    # Stage 7
    stage = '7'
    up2 = K.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(x61)
    diff2 = K.layers.subtract([x31, x32])
    merge2 = K.layers.concatenate([up2, tf.keras.backend.abs(diff2)], axis=3)
    
    conv71 = K.layers.Conv2D(nb_filter[2], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn71 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv72 = K.layers.Conv2D(nb_filter[2], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn72 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop7 = K.layers.Dropout(0.25, name='drop'+ stage)

    x71 = drop7(bn72(conv72(bn71(conv71(merge2)))))
    
    # Stage 8
    stage = '8'
    up3 = K.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(x71)
    diff3 = K.layers.subtract([x21, x22])
    merge3 = K.layers.concatenate([up3, tf.keras.backend.abs(diff3)], axis=3)
    
    conv81 = K.layers.Conv2D(nb_filter[1], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn81 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv82 = K.layers.Conv2D(nb_filter[1], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn82 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop8 = K.layers.Dropout(0.25, name='drop'+ stage)

    x81 = drop8(bn82(conv82(bn81(conv81(merge3)))))

    # Stage 9
    stage = '9'
    up4 = K.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(x81)
    diff4 = K.layers.subtract([x11, x12])
    merge4 = K.layers.concatenate([up4, tf.keras.backend.abs(diff4)], axis=3)
    
    conv91 = K.layers.Conv2D(nb_filter[0], 3, activation='relu', name='conv' + stage + '_1', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn91 = K.layers.BatchNormalization(name='bn' + stage + '_1', axis=3)
    conv92 = K.layers.Conv2D(nb_filter[0], 3, activation='relu', name='conv' + stage + '_2', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))
    bn92 = K.layers.BatchNormalization(name='bn' + stage + '_2', axis=3)
    drop9 = K.layers.Dropout(0.25, name='drop'+ stage)

    x91 = drop9(bn92(conv92(bn91(conv91(merge4)))))

    # Output layer of the U-Net with a softmax activation
    output = K.layers.Conv2D(classes, (1, 1), activation='sigmoid', name='output', padding='same', kernel_initializer='he_normal', kernel_regularizer=K.regularizers.l2(1e-4))(x91)

    model = K.Model(inputs=[input1,input2], outputs=output, name='new_EarlyFusion-UNET')
    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4), loss = 'binary_crossentropy')    
    
    return model
