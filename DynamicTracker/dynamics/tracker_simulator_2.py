import torch
# torch.set_default_dtype(torch.float64)
import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
import cv2
from scipy.io import savemat

# TODO: When debugging
from DynamicTracker.utils_dyn.utils_plots_dynamics import *
from DynamicTracker.dynamics.Tracker_Dynamics_2 import TrackerDyn_2

# TODO: Else
# from utils_dyn.utils_plots_dynamics import *
# from Tracker_Dynamics_2 import TrackerDyn_2


# from SiamMask.utils_dyn.utils_plots_dynamics import *
# from SiamMask.dynamics.Tracker_Dynamics_2 import TrackerDyn_2

img_path = '/Users/marinaalonsopoal/Desktop/Tracking/Datasets/SMOT/acrobats/img/'

# Tracker data
target_sz_dict = '/Users/marinaalonsopoal/Desktop/Objects/positions/target_sz_dict.obj'
target_pos_dict = '/Users/marinaalonsopoal/Desktop/Objects/positions/target_pos_dict.obj'
locs_gt = '/Users/marinaalonsopoal/Desktop/Objects/location_gt'
scores = '/Users/marinaalonsopoal/Desktop/Objects/scores.obj'

with open(locs_gt, 'rb') as f:
    locs_gt = pkl.load(f)  # Manipulated on conv_cand_to_centr.py
with open(target_sz_dict, 'rb') as f:
    target_sz_dict = pkl.load(f)  # Directly from multi_pred.py
with open(target_pos_dict, 'rb') as f:
    target_pos_dict = pkl.load(f)  # Directly from multi_pred.py
with open(scores, 'rb') as f:
    scores = pkl.load(f)  # Directly from multiobject.py

# Parameters
T0 = 8  # System memory
obj = 2
metric = 0  # if 0: JBLD, if 1: JKL
W = 1  # Smoothing window length
slow = False  # If true: Slow(but Precise), if false: Fast
norm = True  # If true: Norm, if false: MSE

metric_name = ['JBLD', 'JKL']
scores = scores[obj]
target_pos = target_pos_dict[obj]  # list of length 154. target_pos[f] is (2,)
target_sz = target_sz_dict[obj]

T = len(target_pos)
tin = 1
tfin = 60  # T - 1

tracker = TrackerDyn_2(T0=T0)

for f in range(1, tfin):  # T
    print('-----Frame:', f, '-----')
    c, pred_pos = tracker.update(target_pos[f-1], target_sz[f-1], scores[f-1])
    if c[0] or c[1]:
        print('PREDICT!')
        # tfin = f + 1
        # break

centroids = tracker.buffer_pos
# savemat('/Users/marinaalonsopoal/Documents/MATLAB/AR_Models/data/c_4.mat', {'data': centroids})
# np.save('/Users/marinaalonsopoal/Desktop/Objects/target_pos_'+str(obj)+'.npy', centroids)

# # Load GT centroids
c_gt = np.load('/Users/marinaalonsopoal/Desktop/Objects/centr_gt_'+str(obj)+'.npy')
c_gt = c_gt[:tfin-1, :]

# plot_jbld_eta_score_2(tracker, c_gt, obj, norm, slow, tin, tfin)


plot_jbld_eta_score_4(tracker, c_gt, obj, norm, slow, tin, tfin)