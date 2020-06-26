# coding: utf-8

"""
Train a CD CNN model for Sentinel-2 datasets

@Author: Tony Di Pilato

Created on Fri Jun 26, 2020
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
from sklearn.utils import class_weight


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=15)
parser.add_argument('--stride', type=int, default=15)
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--batch', '-b', type=int, default=64)
parser.add_argument('--channels', '-ch', type=int, default=13) # Number of channels
parser.add_argument('--loss', '-l', type=str, default='bce', help='bce for classification, mse or any other similar loss function for regression')
parser.add_argument('--model', type=str, default='EF', help='EF only for now')
parser.add_argument('--mode', type=str, default='class', help='class for classification, reg for regression')

args = parser.parse_args()

batch_size = args.batch
img_size = args.size
stride = args.stride
channels = args.channels
epochs = args.epochs
mod = args.model
mode = args.mode
dataset_dir = '../datasets/'
model_dir = '../models/' + mod + '_CNN/'
loss = args.loss
model_name = mod+'_CNN_'+mode+'-'+str(img_size)+'_stride-'+str(stride)+'-'+loss+'_'+str(channels)+'ch'
history_name = model_name + '_history'

model_dir = model_dir + model_name + '/'
hist_dir = model_dir + 'histories/'

os.makedirs(model_dir, exist_ok=True)
os.makedirs(hist_dir, exist_ok=True)

print('*************** Start to read data')
# Read data
inputs = pd.read_hdf(dataset_dir+'onera_'+str(img_size)+'_str-'+str(stride)+'_regression.h5', 'images').values.reshape(-1,img_size,img_size,2*channels)
labels = pd.read_hdf(dataset_dir+'onera_'+str(img_size)+'_str-'+str(stride)+'_regression.h5', 'labels').values.reshape(-1,img_size,img_size,1)
print('*************** Data read')

inputs_pre = inputs[:,:,:,:channels]
inputs_post = inputs[:,:,:,channels:]
inputs = [inputs_pre, inputs_post]

flat_labels = np.reshape(labels, (-1, img_size * img_size))

if(mode=='class'):
    cmat_dir = model_dir + 'confusion_matrix/'
    roc_dir = model_dir + 'roc/'
    os.makedirs(cmat_dir, exist_ok=True)
    os.makedirs(roc_dir, exist_ok=True)
    
    class_labels = np.median(flat_labels, axis=1)

    precision_list = []
    recall_list = []
    f1_list = []
    balanced_acc_list = []

    # Train 5 different models and get the average performance
    kf = KFold(n_splits=5)
    for split, (train, test) in enumerate(kf.split(labels)):

        # Create the model
        model = getattr(cdModels, mod+'_CNN')([img_size,img_size,channels], loss)
        model.summary()
        weights = class_weight.compute_class_weight('balanced', np.unique(class_labels[train]), class_labels[train])
        history = model.fit([inputs_pre[train], inputs_post[train]], class_labels[train], batch_size=batch_size, epochs=epochs, class_weight=weights, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
                
        # Save the history for accuracy/loss plotting
        history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + '-' + str(split) + ".h5", "history", append=False)

        # Save model and weights
        model.save(model_dir + model_name + '-' + str(split) + ".h5")
        print('Trained model saved @ %s ' % model_dir)

        # Now evaluate the model performance
        results = model.predict([inputs_pre[test], inputs_post[test]])

        true_labels = np.reshape(class_labels[test],[-1,1])

        # Define metrics calculate values
        precision = K.metrics.Precision()
        recall = K.metrics.Recall()
        precision.update_state(true_labels, results)
        recall.update_state(true_labels, results)

        precision_value = precision.result().numpy()
        recall_value = recall.result().numpy()
        
        f1_value = (2 * precision_value * recall_value)/(precision_value + recall_value)
        
        balanced_accuracy = balanced_accuracy_score(true_labels, np.rint(results))

        # save metric values
        precision_list.append(precision_value)
        recall_list.append(recall_value)
        f1_list.append(f1_value)
        balanced_acc_list.append(balanced_accuracy)

        # Compute roc_curve and save rates
        fpr, tpr, _ = roc_curve(true_labels, results)
        np.savetxt(roc_dir + 'fpr-'+ str(split) + '.txt', fpr)
        np.savetxt(roc_dir +'tpr-'+ str(split) + '.txt', tpr)
    
        # Compute confusion matrix and save
        cnf_matrix = confusion_matrix(true_labels, np.rint(results), normalize='true')
        np.savetxt(cmat_dir + 'confusion_matrix-'+ str(split) + '.txt', cnf_matrix)

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
    f.write("BalancedAccuracy: %f %f" % (model_balanced_acc[0], model_balanced_acc[1]))
    f.close()

    # If performance is satisfying, deploy model for inference
    if(model_precision[0] > 0.5 and model_recall[0] > 0.6):
        model = getattr(cdModels, mod+'_CNN')([img_size,img_size,channels], loss)
        model.summary()

        weights = class_weight.compute_class_weight('balanced', np.unique(class_labels), class_labels)
        history = model.fit([inputs_pre, inputs_post], class_labels, batch_size=batch_size, epochs=epochs, class_weight=weights, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

        # Save the history for accuracy/loss plotting
        history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + '-final.h5', "history", append=False)

        # Save model and weights
        model.save(model_dir + model_name + '-final.h5')
        print('Deployed model for inference @ %s ' % model_dir)
    else:
        print('The trained models do not satisfy the score requirements')

elif(mode=='reg'):
    reg_dir = model_dir + 'reg/'
    os.makedirs(reg_dir, exist_ok=True)

    counts = [np.unique(flat_labels[k], return_counts=True) for k in range(len(flat_labels))]
    ratios = np.array([float(counts[k][1][1]/(counts[k][1][0]+counts[k][1][1])) if(len(counts[k][1])==2) else(0) for k in range(len(counts))])

    mse_list = []

    # Train 5 different models and get the average performance
    kf = KFold(n_splits=5)
    for split, (train, test) in enumerate(kf.split(labels)):

    # Create the model
        model = getattr(cdModels, mod+'_CNN')([img_size,img_size,channels], loss)
        model.summary()

        history = model.fit([inputs_pre[train], inputs_post[train]], ratios[train], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
                
        # Save the history for accuracy/loss plotting
        history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + '-' + str(split) + ".h5", "history", append=False)

        # Save model and weights
        model.save(model_dir + model_name + '-' + str(split) + ".h5")
        print('Trained model saved @ %s ' % model_dir)

        # Now evaluate the model performance
        results = model.predict([inputs_pre[test], inputs_post[test]])
        
        # Save testdata and predictions for regression plots
        np.savetxt(reg_dir + 'true-'+ str(split) + '.txt', np.squeeze(ratios[test]))
        np.savetxt(reg_dir +'pred-'+ str(split) + '.txt', np.squeeze(results))

        mse = K.metrics.MeanSquaredError()
        mse.update_state(ratios[test], results)
        mse_value = mse.result().numpy()
        mse_list.append(mse_value)
        
    # Save the score of the models
    f = open(model_dir + 'scores.txt',"w+")
    f.write("MSE: %s" % mse_list)
    f.close()
 
    mean_mse = np.mean(mse_list)
    
    # If performance is satisfying, deploy model for inference
    if(mean_mse < 0.005):
        model = getattr(cdModels, mod+'_CNN')([img_size,img_size,channels], loss)
        model.summary()

        history = model.fit([inputs_pre[train], inputs_post[train]], ratios[train], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[K.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
                
        # Save the history for loss plotting
        history_save = pd.DataFrame(history.history).to_hdf(hist_dir + history_name + '-final.h5', "history", append=False)

        # Save model and weights
        model.save(model_dir + model_name + '-final.h5')
        print('Deployed model for inference @ %s ' % model_dir)
    else:
        print('The trained models do not satisfy the score requirements')
else:
    print('Mode not accepted (class or reg only)')