import torch
import pickle as pkl
import numpy as np
from TrackerDynBoxes import TrackerDynBoxes
from utils_visualization import *
from utils_general import *
import matplotlib.pyplot as plt


def plot_centr_gt_pred_tog(gt, pred):
    # Parameters
    col = ['g', 'm', 'y', 'b', 'r']
    s_gt = 50
    s_pred = 25

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('Coordinate x')
    ax2.set_title('Coordinate y')
    ax2.set_xlabel('frame')
    ax1.set_ylabel('pixel')
    ax2.set_ylabel('pixel')

    f = np.arange(gt[1].shape[0])

    for key, value in gt.items():
        l1 = ax1.scatter(f, value[:, 0], c=col[key - 1], s=s_gt, alpha=0.2)
        ax2.scatter(f, value[:, 1], c=col[key - 1], s=s_gt, alpha=0.2)

        value_pred = pred[key]
        l2 = ax1.scatter(f, value_pred[:, 0], c=col[key - 1], s=s_pred, edgecolors='k', alpha=1)
        ax2.scatter(f, value_pred[:, 1], c=col[key - 1], s=s_pred, edgecolors='k', alpha=1)

        if key == 1:
            l1.set_label('Ground Truth')
            l2.set_label('Predictions')

    ax1.legend()
    plt.show()


def plot_centr_gt_pred_sep(gt, pred):
    # Parameters
    col = ['g', 'm', 'y', 'b', 'r']
    s_gt = 50
    s_pred = 25

    fig, ax = plt.subplots(2, 5)
    ax[0, 0].set_ylabel('Coordinate x')
    ax[1, 0].set_ylabel('Coordinate y')

    f = np.arange(gt[1].shape[0])

    for key, value in gt.items():
        ax[0, key - 1].set_title('Object ' + str(key))

        ax[1, key - 1].set_xlabel('frame')
        l_gt = ax[0, key - 1].scatter(f, value[:, 0], c=col[key - 1], s=s_gt, alpha=0.2)
        ax[1, key - 1].scatter(f, value[:, 1], c=col[key - 1], s=s_gt, alpha=0.2)

        value_pred = pred[key]
        l_pred = ax[0, key - 1].scatter(f, value_pred[:, 0], c=col[key - 1], s=s_pred, edgecolors='k', alpha=1)
        ax[1, key - 1].scatter(f, value_pred[:, 1], c=col[key - 1], s=s_pred, edgecolors='k', alpha=1)

        l_gt.set_label('GT')
        l_pred.set_label('Pred.')
        ax[0, key - 1].legend()

        # if key == 1:
        #     l1.set_label('Ground Truth')
        #     l2.set_label('Predictions')

    # ax1.legend()
    plt.show()


def compute_metric_centroids(gt, pred):
    # gt and pred must be dictionaries with the object number as keys and a (n_frmes, 2) array as values
    dif = np.zeros((2, len(gt)))
    for key, value in gt.items():
        value_pred = pred[key]
        dif[0, key-1] = int(np.mean((np.absolute(value[:, 0] - value_pred[:, 0]))))
        dif[1, key-1] = int(np.mean((np.absolute(value[:, 1] - value_pred[:, 1]))))
    print('Dif for (coord, obj):\n', dif)
    # dif_total = int(np.mean(dif_obs))  # mean pixel error for coordinate, object and frame
    # print('Dif total:', dif_total)
    return int(np.mean(dif))


# Tracker data
pred = '/Users/marinaalonsopoal/Desktop/dict_pred.obj'
gt = '/Users/marinaalonsopoal/Desktop/centr_gt'
# NOTE: Per passar el gt a centroides, ho he fet al COLAB
with open(pred, 'rb') as f:
    pred = pkl.load(f)
with open(gt, 'rb') as f:
    gt = pkl.load(f)


plot_centr_gt_pred_sep(gt, pred)
ce = compute_metric_centroids(gt, pred)
print('ce:', ce)



