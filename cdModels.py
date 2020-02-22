# coding: utf-8

"""
Change Detection models Sentinel-2 datasets

@Author: Tony Di Pilato

Created on Wed Feb 19, 2020
"""

from tensorflow import keras as K


def UNet_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3, mode='None'):   
    x = K.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_1', kernel_initializer='he_normal', padding='same', kernel_regularizer=K.regularizers.l2(1e-4))(input_tensor)
#    x0 = x
#    x = BatchNormalization(name='bn' + stage + '_1')(x)
    x = K.layers.Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_2', kernel_initializer='he_normal', padding='same', kernel_regularizer=K.regularizers.l2(1e-4))(x)
    x = K.layers.BatchNormalization(name='bn' + stage)(x)
#    x = BatchNormalization(name='bn' + stage + '_2')(x)
    if mode == 'residual':
        x = K.Add(name='resi' + stage)([x, x0])
    return x


def EF_UNet(input_shape, classes=1):
    mode = 'None'
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    # Left side of the U-Net
    inputs = K.Input(shape=input_shape, name='input')

    conv1 = UNet_ConvUnit(inputs, stage='1', nb_filter=nb_filter[0], mode=mode)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UNet_ConvUnit(pool1, stage='2', nb_filter=nb_filter[1], mode=mode)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UNet_ConvUnit(pool2, stage='3', nb_filter=nb_filter[2], mode=mode)
    pool3 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = UNet_ConvUnit(pool3, stage='4', nb_filter=nb_filter[3], mode=mode)
    pool4 = K.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottom of the U-Net
    conv5 = UNet_ConvUnit(pool4, stage='5', nb_filter=nb_filter[4], mode=mode)
    
    # Right side of the U-Net
    up1 = K.layers.Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(conv5)
    merge1 = K.layers.concatenate([conv4,up1], axis=bn_axis)
    conv6 = UNet_ConvUnit(merge1, stage='6', nb_filter=nb_filter[3], mode=mode)

    up2 = K.layers.Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(conv6)
    merge2 = K.layers.concatenate([conv3,up2], axis=bn_axis)
    conv7 = UNet_ConvUnit(merge2, stage='7', nb_filter=nb_filter[2], mode=mode)

    up3 = K.layers.Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(conv7)
    merge3 = K.layers.concatenate([conv2,up3], axis=bn_axis)
    conv8 = UNet_ConvUnit(merge3, stage='8', nb_filter=nb_filter[1], mode=mode)
    
    up4 = K.layers.Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(conv8)
    merge4 = K.layers.concatenate([conv1,up4], axis=bn_axis)
    conv9 = UNet_ConvUnit(merge4, stage='9', nb_filter=nb_filter[0], mode=mode)

    # Output layer of the U-Net with a softmax activation
    output = K.layers.Conv2D(classes, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=K.regularizers.l2(1e-4))(conv9)

    model = K.Model(inputs=inputs, outputs=output, name='UNET')

    model.compile(optimizer=K.optimizers.Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])    
    
    return model

