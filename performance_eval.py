# coding: utf-8

"""
Evaluate performance of a CD UNet/UNet++ trained model for Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Mon Feb 24, 2020
"""


import os
import cdUtils
import numpy as np
from osgeo import gdal
import tensorflow as tf
from tensorflow import keras as K
import pandas as pd
import cdModels
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, auc, confusion_matrix, roc_curve
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('-cpt', type=int, default=600) # Number of crops per tiff
parser.add_argument('--batch', '-b', type=int, default=32)
parser.add_argument('--channels', '-ch', type=int, default=13) # Number of channels
parser.add_argument('--model', type=str, default='EF', help='EF, Siam or SiamDiff')
parser.add_argument('--loss', '-l', type=str, default='bce', help='bce or bced or dice')
args = parser.parse_args()

batch_size = args.batch
img_size = args.size
channels = args.channels
aug = args.augmentation
cpt = args.cpt
mod = args.model
classes = 1
dataset_dir = '../CD_wOneraDataset/OneraDataset_Images/'
labels_dir = '../CD_wOneraDataset/OneraDataset_TrainLabels/'
model_dir = 'models/' + mod + '/'
hist_dir = 'histories/' + mod + '/'
plot_dir = 'plots/' + mod + '/'
score_dir = 'scores/' + mod + '/'
loss = args.loss

test_name = mod+'_'+str(img_size)+'_aug-'+str(cpt)+'-'+loss
model_name = test_name+'_'+str(channels)+'channels'


plot_dir = plot_dir+test_name+'/'
score_dir = score_dir+test_name+'/'
history_name = model_name + '_history'
class_names = ['unchange', 'change']

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(score_dir, exist_ok=True)

# Plot the loss function during training and validation steps
history = pd.read_hdf(hist_dir + history_name + ".h5", "history")
n_epochs = len(history)
n_epochs = np.arange(1, n_epochs+1)

val_loss = history['val_loss'].values
train_loss = history['loss'].values

plt.figure()
plt.plot(n_epochs, train_loss, 'royalblue', label='Training')
plt.plot(n_epochs, val_loss, 'orangered', label='Validation')
plt.title(model_name + ' loss', y=1.04)
plt.grid(linestyle=':')
plt.xlabel('Epoch', labelpad=8, fontsize=14)
plt.ylabel('Loss', labelpad=10, fontsize=14)
plt.legend(loc='upper right')
plt.savefig(plot_dir + model_name + '_loss.pdf', format='pdf')

# Get the list of folders to open to get rasters
f = open(dataset_dir + 'train.txt', 'r')
folders = f.read().split(',')
f.close()

train_images = []
num_crops = []
padded_shapes = []
train_labels = []
unpadded_shapes = []

for f in folders:
    
    raster1 = cdUtils.build_raster(dataset_dir + f + '/imgs_1_rect/', channels)
    raster2 = cdUtils.build_raster(dataset_dir + f + '/imgs_2_rect/', channels)
    raster = np.concatenate((raster1,raster2), axis=2)
    padded_raster = cdUtils.pad(raster, img_size)
    shape = (padded_raster.shape[0], padded_raster.shape[1], classes)
    padded_shapes.append(shape)
    crops = cdUtils.crop(padded_raster, img_size, img_size)
    num_crops.append(len(crops))
    train_images = train_images + crops

    #Read change maps to get the ground truths
    cm = gdal.Open(labels_dir + f + '/cm/' + f + '-cm.tif').ReadAsArray()
    cm = np.expand_dims(cm, axis=2)
    cm -= 1 # the change map has values 1 for unchange and 2 for change ---> scale back to 0 and 1
    unpadded_shapes.append(cm.shape)
    cm = cm.astype('float32')
    cm = cm.flatten()
    train_labels.append(cm)

# Create inputs for the Neural Network
inputs = np.asarray(train_images, dtype='float32')

if(mod=='Siam' or mod=='SiamDiff' or mod=='new_Siam'):
    inputs_1 = inputs[:,:,:,:channels]
    inputs_2 = inputs[:,:,:,channels:]
    inputs = [inputs_1, inputs_2]

# Load the model
if(loss=='bced' or loss=='dice'):
    model = K.models.load_model(model_dir + model_name + '.h5', custom_objects={'weighted_bce_dice_loss': cdModels.weighted_bce_dice_loss, 'dice_coef_loss':cdModels.dice_coef_loss})
else:
    model = K.models.load_model(model_dir + model_name + '.h5')    
model.summary()

# Perform inference
results = model.predict(inputs)

# Build unpadded change maps 
index = 0
y_pred = []

for i in range(len(folders)):
    crops = num_crops[i]
    padded_cm = cdUtils.uncrop(padded_shapes[i], results[index:index+crops], img_size, img_size)
    cm = cdUtils.unpad(unpadded_shapes[i], padded_cm)
    cm = cm.flatten()
    y_pred.append(cm)
    index += crops

# Flatten results
y_pred = [item for sublist in y_pred for item in sublist]
y_true = [item for sublist in train_labels for item in sublist]

# Define metrics calculate values
precision = K.metrics.Precision()
recall = K.metrics.Recall()
precision.update_state(y_true, y_pred)
recall.update_state(y_true, y_pred)

f1 = (2 * precision.result().numpy() * recall.result().numpy())/(precision.result().numpy() + recall.result().numpy())
y_pred_r = np.rint(y_pred)  
bacc = balanced_accuracy_score(y_true, y_pred_r)

# Save scores
f = open(score_dir + model_name + '_scores.txt',"w+")

f.write("Precision: %f\n" % precision.result().numpy())
f.write("Recall: %f\n" % recall.result().numpy())
f.write("F1: %f\n" % f1)
f.write("Balanced Accuracy: %f\n" % bacc)

f.close()

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=13, y=1.04)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel('True class', labelpad=10, fontsize=13)
    plt.xlabel('Predicted class', labelpad=10, fontsize=13)
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred_r, normalize='true')
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig(plot_dir + model_name + '_cm.pdf', format='pdf')
plt.show()


# ROC curve
fpr = [] ###### NOT NECESSARY! BELOW ROC_CURVE READS AS ARRAYS 
tpr = [] ###### TO BE LOADED WITH np.loadtxt

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='orangered', lw=1.5, label=model_name+' (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='royalblue', lw=1.5, linestyle='--')
plt.plot([0, 1], [1, 1], color='black', lw=0.7, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(plot_dir + model_name + '_roc.pdf', format='pdf')
plt.show()

# Finally, save trp and fpr to file for roc curve comparison  

rates = ['tpr', 'fpr']
datas = np.stack((tpr,fpr))
df = pd.DataFrame(datas, index = rates)
df.to_hdf(score_dir + model_name + '_rates.h5',"rates",complevel=0)
