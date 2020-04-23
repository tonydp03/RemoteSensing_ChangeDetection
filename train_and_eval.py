# coding: utf-8

"""
Train a CD UNet/UNet++ model for Sentinel-2 datasets

@Author: Tony Di Pilato

Created on Wed Feb 19, 2020
"""


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras as K
import cdUtils
from osgeo import gdal
import pandas as pd
import cdModels
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_curve


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--batch', '-b', type=int, default=64)
parser.add_argument('-cpt', type=int, default=500) # Number of crops per tiff
parser.add_argument('--channels', '-ch', type=int, default=13) # Number of channels
parser.add_argument('--loss', '-l', type=str, default='bce', help='bce, bced or dice')
parser.add_argument('--model', type=str, default='EF', help='EF, Siam or SiamDiff')

args = parser.parse_args()

batch_size = args.batch
img_size = args.size
channels = args.channels
classes = 1
epochs = args.epochs
cpt = args.cpt
mod = args.model
dataset_dir = 'datasets/'
model_dir = 'models/' + mod + '/'
loss = args.loss

model_name = mod+'_'+str(img_size)+'_cpt-'+str(cpt)+'-'+loss+'_'+str(channels)+'channels'
history_name = model_name + '_history'

model_dir = model_dir + model_name + '/'
hist_dir = model_dir + 'histories/'

os.makedirs(hist_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Read data
inputs_pre = pd.read_hdf(dataset_dir+'onera_'+str(img_size)+'_cpt-'+str(cpt)+'.h5', 'images_pre').values.reshape(-1,img_size,img_size,channels)
inputs_post = pd.read_hdf(dataset_dir+'onera_'+str(img_size)+'_cpt-'+str(cpt)+'.h5', 'images_post').values.reshape(-1,img_size,img_size,channels)
labels = pd.read_hdf(dataset_dir+'onera_'+str(img_size)+'_cpt-'+str(cpt)+'.h5', 'labels').values.reshape(-1,img_size,img_size,1)

inputs_pre = inputs_pre.astype('float32')
inputs_post = inputs_post.astype('float32')
labels = labels.astype('float32')

# Now shuffle data to split them better
inputs_pre, inputs_post, labels = shuffle(inputs_pre, inputs_post, labels)

precision_list = []
recall_list = []
f1_list = []
balanced_acc_list = []

# Train 5 different models and get the average performance
kf = KFold(n_splits=5)
for split, (train, test) in enumerate(kf.split(labels)):
# Create the model
    model = getattr(cdModels, mod+'_UNet')([img_size,img_size,channels], classes, loss)
    model.summary()

    if(loss=='bce'):
        # Compute class weights
        flat_labels = np.reshape(labels[train],[-1])
        weights = class_weight.compute_class_weight('balanced', np.unique(flat_labels), flat_labels)
        history = model.fit([inputs_pre[train], inputs_post[train]], labels[train], batch_size=batch_size, epochs=epochs, class_weight=weights, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
    else:
        history = model.fit([inputs_pre[train], inputs_post[train]], labels[train], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

    # Save the history for accuracy/loss plotting
    history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + '-' + str(split) + ".h5", "history", append=False)

    # Save model and weights
    model.save(model_dir + model_name + '-' + str(split) + ".h5")
    print('Trained model saved @ %s ' % model_dir)

    # Now evaluate the model performance
    results = model.predict([inputs_pre[test], inputs_post[test]])

    # Define metrics calculate values
    precision = K.metrics.Precision()
    recall = K.metrics.Recall()
    precision.update_state(labels[test], results)
    recall.update_state(labels[test], results)

    precision_value = precision.result().numpy()
    recall_value = recall.result().numpy()
    f1_value = (2 * precision_value * recall_value)/(precision_value + recall_value)
    
    balanced_accuracy = balanced_accuracy_score(labels[test].reshape(-1), np.rint(results.reshape(-1)))

    # save metric values
    precision_list.append(precision_value)
    recall_list.append(recall_value)
    f1_list.append(f1_value)
    balanced_acc_list.append(balanced_accuracy)

    # Compute roc_curve and save rates
    fpr, tpr, _ = roc_curve(np.reshape(labels[test], -1), np.reshape(results, -1))
    np.savetxt(model_dir + 'fpr-'+ str(split) + '.txt', fpr)
    np.savetxt(model_dir +'tpr-'+ str(split) + '.txt', tpr)
 
    # Compute confusion matrix and save
    cnf_matrix = confusion_matrix(np.reshape(labels[test], -1), np.rint(np.reshape(results, -1)), normalize='true')
    np.savetxt(model_dir + 'confusion_matrix-'+ str(split) + '.txt', cnf_matrix)


# Save the scores of the models
f = open(model_dir + 'total_scores.txt',"w+")
f.write("Precision: %s\n" % precision_list)
f.write("Recall: %s\n" % recall_list)
f.write("F1: %s\n" % f1_list)
f.write("BalancedAccuracy: %s" % balanced_acc_list)
f.close()

# Calculate average performance of the models
model_precision = [np.mean(precision_list), np.std(precision_list)] 
model_recall = [np.mean(recall_list), np.std(recall_list)]
model_f1 = [np.mean(f1_list), np.std(f1_list)]
model_balanced_acc = [np.mean(balanced_acc_list), np.std(balanced_acc_list)]

# Save the average scores of the models
f = open(model_dir + 'scores.txt',"w+")
f.write("Precision: %f %f\n" % (model_precision[0], model_precision[1]))
f.write("Recall: %f %f\n" % (model_recall[0], model_recall[1]))
f.write("F1: %f %f\n" % (model_f1[0], model_f1[1]))
f.write("BalancedAccuracy: %f %f\n" % (model_balanced_acc[0], model_balanced_acc[1]))
f.close()

# If performance is satisfying, deploy model for inference
if(model_precision[0] > 0.5 and model_recall[0] > 0.6):
    model = getattr(cdModels, mod+'_UNet')([img_size,img_size,channels], classes, loss)
    model.summary()

    if(loss=='bce'):
        # Compute class weights
        flat_labels = np.reshape(labels,[-1])
        weights = class_weight.compute_class_weight('balanced', np.unique(flat_labels), flat_labels)
        history = model.fit([inputs_pre, inputs_post], labels, batch_size=batch_size, epochs=epochs, class_weight=weights, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
    else:
        history = model.fit([inputs_pre, inputs_post], labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

    # Save the history for accuracy/loss plotting
    history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + '-final.h5', "history", append=False)

    # Save model and weights
    model.save(model_dir + model_name + '-final.h5')
    print('Deployed model for inference @ %s ' % model_dir)

else:
    print('The trained models do not satisfy the score requirements')

print('Training and evaluation phases over')