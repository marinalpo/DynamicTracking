import torch
import pickle as pkl
import numpy as np

from utils_visualization import *
from utils_general import *
import matplotlib.pyplot as plt

device = torch.device('cpu')
# device = torch.device('cuda:0')

# Parameters
eps = 1e-1  # Gram Matrix noise
T0 = 6  # Past temporal window
T = 2  # Future temporal window
W = 1  # Smoothing window size - If W=1, no smoothing is performed
s = int(np.ceil(W/2)-1)
coordinate = 0  # coordinate 0: x, 1: y

col = ['g', 'm', 'y', 'b', 'r']

# Tracker data
directory = '/Users/marinaalonsopoal/Desktop/centroids_SMOT_acrobats.obj'
gt = '/Users/marinaalonsopoal/Desktop/centr_gt'
# NOTE: Per passar el gt a centroides, ho he fet al COLAB

with open(directory, 'rb') as f:
    data = pkl.load(f)

with open(gt, 'rb') as f:
    gt = pkl.load(f)


for key, value in gt.items():
    print('key:', key)
    frames = np.arange(0, value.shape[0])



for key, value in data.items():
    # key: Object ID, value: rboxes [list] of length (number of frames)
    print('-----------------ObjectID:', key, '--------------------')
    for f, cands in enumerate(value):
        # f: frame, cands: candidates info [list] of length (number of candidates)
        # print('Frame:', f)  # Actually, frame f+1
        for candID, info in enumerate(cands):
            # print('candID:', candID)
            location = info[0]  # numpy.ndarray(4,2)
            score = info[1]
            c = np.average(location, axis=0)
            plt.scatter(f, c[0], c=col[key-1], s=score*50, edgecolors='k', alpha=0.3)
plt.show()