# coding: utf-8

"""
Evaluate performance of a CD UNet/UNet++ trained model for Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Sat May 2, 2020
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
from sklearn.metrics import auc
import itertools
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('-cpt', type=int, default=500) # Number of crops per tiff
parser.add_argument('--channels', '-ch', type=int, default=13) # Number of channels
parser.add_argument('--model', type=str, default='EF', help='EF, Siam or SiamDiff')
parser.add_argument('--loss', '-l', type=str, default='bce', help='bce or bced or dice')
args = parser.parse_args()

img_size = args.size
channels = args.channels
cpt = args.cpt
mod = args.model
classes = 1
model_dir = 'models/' + mod + '/'
loss = args.loss

model_name = mod+'_'+str(img_size)+'_cpt-'+str(cpt)+'-'+loss+'_'+str(channels)+'channels'
history_name = model_name + '_history'
model_dir = model_dir + model_name + '/'
hist_dir = model_dir + 'histories/'
roc_dir = model_dir + 'roc/'
cmat_dir = model_dir + 'confusion_matrix/'


histories = [f for f in os.listdir(hist_dir) if(f.endswith('.h5'))]

train_loss = []
val_loss = []
epochs = []
hist_num = []

# Plot the loss function during training and validation steps
for hist in histories:
    hist_name = hist.rsplit(('-'),1)
    hist_name = hist_name[1].rsplit(('.'),1)
    history = pd.read_hdf(hist_dir + hist, "history")
    n_epochs = np.arange(1, len(history)+1)
    if (hist_name[0] != 'final'):
        val_loss.append(history['val_loss'].values)
        train_loss.append(history['loss'].values)
        hist_num.append(hist_name[0])
        epochs.append(n_epochs)
    else:
        val_final = history['val_loss'].values
        train_final = history['loss'].values
        epochs_final = n_epochs

# Sorting model performances by their model number
train_loss = [x for _,x in sorted(zip(hist_num,train_loss))]
val_loss = [x for _,x in sorted(zip(hist_num,val_loss))]
epochs = [x for _,x in sorted(zip(hist_num,epochs))]


# Plot training losses
plt.figure()
for i in range(len(hist_num)):
    plt.plot(epochs[i], train_loss[i], color='C'+str(i), lw=1, label='model %d' % i)
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.grid(ls=':')
plt.title(model_name + ' - train loss')
plt.legend(loc="upper right", fontsize=9.3)
plt.savefig(hist_dir + 'train_losses.pdf', format='pdf')

# Plot validation losses
plt.figure()
for i in range(len(hist_num)):
    plt.plot(epochs[i], val_loss[i], color='C'+str(i), lw=1, label='model %d' % i)
plt.xlabel('Epoch')
plt.ylabel('Validation loss')
plt.grid(ls=':')
plt.title(model_name + ' - validation loss')
plt.legend(loc="upper right", fontsize=9.3)
plt.savefig(hist_dir + 'val_losses.pdf', format='pdf')

try:
    # Plot training and validation losses for final model 
    plt.figure()
    plt.plot(epochs_final, train_final, color='royalblue', label='Training')
    plt.plot(epochs_final, val_final, color='orangered', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(ls=':')
    plt.title(model_name + ' - final loss')
    plt.legend(loc="upper right", fontsize=9.3)
    plt.savefig(hist_dir + 'final_losses.pdf', format='pdf')
except:
    print('The model did not satisfy the performance requirements')


# Plot the ROC curves for all the models
plt.figure()
for i in range(5):
    fpr = np.loadtxt(roc_dir + 'fpr-'+ str(i)+'.txt')
    tpr = np.loadtxt(roc_dir + 'tpr-'+ str(i)+'.txt')
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='C'+str(i), lw=1, label='model %d (AUC = %0.3f)' % (i, roc_auc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(model_name + ' - roc curves')
plt.grid(ls=':')
plt.legend(loc="lower right")
plt.savefig(roc_dir + 'roc_curves.pdf', format='pdf')
plt.show()

# Function to plot the confusion matrix
def plot_confusion_matrix(axis, title, cm, true_class, pred_class, cmap=plt.cm.YlGnBu):
    
    axis.imshow(cm, interpolation='nearest', cmap=cmap)
    axis.set_title(title, fontsize=10)
    tick_marks = np.arange(len(true_class))
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(pred_class, fontsize=10)
    axis.set_yticks(tick_marks)
    axis.set_yticklabels(true_class, fontsize=10)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axis.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=10)


# Plot normalized confusion matrices
true_class = ['TU', 'TC']
pred_class = ['PU', 'PC']

cm_list = []
for i in range(5):
    cm_list.append(np.loadtxt(cmat_dir + 'confusion_matrix-'+ str(i)+'.txt'))

np.set_printoptions(precision=2)

fig = plt.figure()
fig.suptitle(model_name + ' - confusion matrix', fontsize=12, y=0.96)

gs = fig.add_gridspec(nrows=2, ncols=6, wspace=2)
ax0 = fig.add_subplot(gs[:1,:2])
ax1 = fig.add_subplot(gs[:1,2:4])
ax2 = fig.add_subplot(gs[:1,4:])
ax3 = fig.add_subplot(gs[1:,1:3])
ax4 = fig.add_subplot(gs[1:,3:5])
ax_list =[ax0, ax1, ax2, ax3, ax4]
for i in range(len(ax_list)): 
    plot_confusion_matrix(ax_list[i], 'Model '+ str(i), cm_list[i], true_class, pred_class)

textstr = '\n'.join((
    r'TU = True Unchange',
    r'TC = True Change',
    r'PU = Predicted Unchange',
    r'PC = Predicted Change'))

fig.text(0.78, 0.12, textstr, fontsize=7, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='grey', edgecolor='grey', alpha=0.3))
fig.savefig(cmat_dir + 'cm.pdf', format='pdf')
fig.show()

# Plot the total scores
fp = open(model_dir+'total_scores.txt')
total_scores = []
score_names = []
for line in fp:
    splits = line.split(':')
    score = splits[1].replace("\n", "")
    score = [float(x) for x in score[2:-1].split(', ')]
    total_scores.append(score)
    score_names.append(splits[0])
fp.close()

model_performance = []

for i in range(len(total_scores[0])):
    performance = [x[i] for x in total_scores]
    model_performance.append(performance)

fig = plt.figure()
bar_size = 0.30
padding = 0.30
y_pos = np.arange(len(score_names)) * (bar_size * 5 + padding)
for i in range(5):
    plt.barh(y_pos+(i-2)*bar_size, model_performance[i], align='center', height=bar_size, color='C'+str(i), label='model %d' %i)

plt.xlim(0, 1)
plt.yticks(y_pos, score_names)
plt.gca().invert_yaxis()
plt.xlabel('Value')
plt.title(model_name + ' - scores')

plt.legend(loc='right', fontsize=7)
plt.tight_layout()
fig.savefig(model_dir + 'total_performance.pdf', format='pdf')
fig.show()


# Plot the mean scores
fp = open(model_dir+'scores.txt')
mean_scores = []
err_scores = []
score_names = []
for line in fp:
    splits = line.split(':')
    score = splits[1].replace("\n", "")
    score, err = score[1:].split(' ')
    mean_scores.append(float(score))
    err_scores.append(float(err))
    score_names.append(splits[0])
fp.close()

percentages = ['{:.2f}'.format(k*100) for k in mean_scores]
fig = plt.figure()
y_pos = np.arange(len(score_names))
plt.barh(y_pos, mean_scores, xerr=err_scores, error_kw= dict(lw=1, capsize=8, capthick=1), height= 0.5, align='center', color='seagreen')
for i in range(len(mean_scores)):
    plt.text(0.03, y_pos[i]+0.03, percentages[i]+'%', color='white', weight='bold', fontsize=8)

plt.xlim(0, 1)
plt.yticks(y_pos, score_names)
plt.gca().invert_yaxis()
plt.xlabel('Value')
plt.title(model_name + ' - mean scores')

plt.tight_layout()
plt.savefig(model_dir + 'mean_performance.pdf', format='pdf')
plt.show()

