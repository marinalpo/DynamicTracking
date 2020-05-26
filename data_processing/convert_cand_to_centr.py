import torch
import pickle as pkl
import numpy as np
from TrackerDynBoxes import TrackerDynBoxes
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

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].set_title('Ground Truth centroid coordinate x')
# ax[0, 1].set_title('Ground Truth centroid coordinate y')
# ax[1, 0].set_title('Candidates centroid coordinate x')
# ax[1, 1].set_title('Candidates centroid coordinate y')
#
# ax[0, 0].set_ylabel('pixel')
#
# ax[1, 0].set_xlabel('frame')
# ax[1, 0].set_ylabel('pixel')
# ax[1, 1].set_xlabel('frame')
#
# for key, value in gt.items():
#     # print('key:', key)
#     # print('type value:', type(value))
#     # print('shape value:', value.shape)
#     frames = np.arange(0, value.shape[0])
#     ax[0, 0].scatter(frames, value[:, 0], c=col[key-1], edgecolors='k', alpha=0.5)
#     ax[0, 1].scatter(frames, value[:, 1], c=col[key - 1], edgecolors='k', alpha=0.5)

dict_pred = {}
for key, value in data.items():
    # key: Object ID, value: rboxes [list] of length (number of frames)
    print('----------------- ObjectID:', key, '--------------------')
    centr = np.zeros((154, 2))
    for f, cands in enumerate(value):
        # f: frame, cands: candidates info [list] of length (number of candidates)
        print('Frame:', f)  # Actually, frame f+1
        for candID, info in enumerate(cands):
            print('info:', info)
            if candID == 0 and f < 154:
                # location = info[0]  # numpy.ndarray(4,2)
                # score = info[1]
                # c = np.average(location, axis=0)
                c = info[0]
                centr[f, :] = c
            # ax[1, 0].scatter(f, c[0], c=col[key-1], s=score*50, edgecolors='k', alpha=0.3)
            # ax[1, 1].scatter(f, c[1], c=col[key - 1], s=score * 50, edgecolors='k', alpha=0.3)
    dict_pred[key] = centr
plt.show()

for key, value in dict_pred.items():
    print('key:', key)
    print('value:', value)


with open('/Users/marinaalonsopoal/Desktop/dict_pred.obj', 'wb') as fil:
    pkl.dump(dict_pred, fil)