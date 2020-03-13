import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from tqdm import tqdm
from mpl_toolkits import mplot3d
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
args = parser.parse_args()

img_size = args.size
score_dir = 'scores/'
plot_dir = 'plots/test/'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# ROC comparison
files = [f for f in os.listdir(score_dir) if (str(img_size) in f and f.endswith("h5"))]
tpr = []
fpr = []
roc_auc = []
models = []

for(i, name) in zip(range(len(files)), tqdm(files)):
    print("Reading file", i, name)
    data = pd.read_hdf(score_dir + name)
    tpr.append(data.loc['tpr'].values)
    fpr.append(data.loc['fpr'].values)
    roc_auc.append(auc(fpr[i], tpr[i]))
    model = name.rsplit('_',2)
    model_id = model[1]
    model_name = model[0]
    models.append(model_id)

plot_dir = plot_dir + model_name + '/'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# plot ROC curves
plt.figure()
for i in range(len(tpr)):
    plt.plot(fpr[i], tpr[i], color='C'+str(i), lw=1.5, label=models[i]+' (AUC = %0.2f)' % roc_auc[i])
plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')
plt.plot([0, 1], [1, 1], color='black', lw=0.7, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(model_name)
plt.legend(loc="lower right", fontsize=9.3)
plt.savefig(plot_dir + model_name + '_roc_comparison.pdf', format='pdf')


# Score comparison
files = [f for f in os.listdir(score_dir) if (model_name in f and f.endswith("txt") and not f.startswith("."))]
scores = []
labels = []

for(i, name) in zip(range(len(files)), tqdm(files)):
    print("Reading file", i, name)
    fp = open(score_dir+name)
    file_scores = []
    for line in fp:
        splits = line.strip().split(':')
        file_scores.append(float(splits[1]))
    scores.append(file_scores)
    model = name.rsplit('_',2)
    labels.append(model[1])
    model_name = model[0]

# plot scores 3D
ax = plt.axes(projection='3d')
color = ['C'+str(i) for i in range(len(scores))]

for i in range(len(scores)):
    ax.scatter3D(scores[i][0], scores[i][1], scores[i][2], c=color[i], s=70, label=labels[i])

ax.set_xlabel('Precision')
ax.set_ylabel('Recall')
ax.set_zlabel('Balanced Accuracy')
plt.tight_layout()
plt.title(model_name, fontsize=11, pad=0)
ax.legend(loc='upper left')
plt.savefig(plot_dir + model_name + '_3Dscore_comparison.pdf', format='pdf')

# plot scores 2D
ax = plt.axes()
for i in range(len(scores)):
    ax.scatter(scores[i][2], scores[i][3], c=color[i], s=70, label=labels[i])
ax.set_xlabel('F1 score')
ax.set_ylabel('Balanced Accuracy')
plt.tight_layout()
plt.title(model_name, fontsize=11)
ax.legend(loc='upper left')
ax.grid(ls=':')
ax.set_axisbelow(True)
plt.savefig(plot_dir + model_name + '_2Dscore_comparison.pdf', format='pdf')