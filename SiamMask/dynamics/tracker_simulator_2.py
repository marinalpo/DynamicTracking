import torch
# torch.set_default_dtype(torch.float64)
import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt
# from utils_dyn.utils_plots_dynamics import *
# from Tracker_Dynamics_2 import TrackerDyn_2
# from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
import cv2
from scipy.io import savemat



from SiamMask.utils_dyn.utils_plots_dynamics import *
from SiamMask.dynamics.Tracker_Dynamics_2 import TrackerDyn_2

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
T0 = 11  # System memory
R = 5
eps = 1  # Noise variance
obj = 2
metric = 0  # if 0: JBLD, if 1: JKL
W = 3  # Smoothing window length
slow = False  # If true: Slow(but Precise), if false: Fast
norm = True  # If true: Norm, if false: MSE

metric_name = ['JBLD', 'JKL']
scores = scores[obj]
target_pos = target_pos_dict[obj]  # list of length 154. target_pod[f] is (2,)
target_sz = target_sz_dict[obj]


T = len(target_pos)
tin = 1
tfin = 154  #len(loc)

tracker = TrackerDyn_2(T0=T0, R=R,  W=W, noise=eps, metric=metric, slow=slow, norm=norm)
for f in range(1, tfin):  # T
    print('-----Frame:', f, '-----')
    c, pred_pos = tracker.update(target_pos[f], target_sz[f], scores[f])
    if c:
        print('predict!', pred_pos)
        # tfin = f + 1
        # break
centroids = tracker.buffer_pos
savemat('/Users/marinaalonsopoal/Documents/MATLAB/AR_Models/data/c_2.mat', {'data': centroids})

plot_jbld_eta_score_2(tracker, obj, norm, slow, tin, tfin)

# f = np.arange(tin, tfin)
# w = tracker.buffer_sz[:, 0]
# plt.scatter(f, w)
# plt.show()
# plot_jbld_eta_score(tracker_pred, tracker_gt, obj, norm, slow, tin, tfin)
# f = 36
# target_pos = tracker.buffer_pos[f, :]
# target_sz = tracker.buffer_sz[f, :]
# print('target pos:', target_pos)
# print('target sz:', target_sz)
# location = cxy_wh_2_rect(target_pos, target_sz)
# print('location:', location)
# rbox_in_img = np.array([[location[0], location[1]],
#                         [location[0] + location[2], location[1]],
#                         [location[0] + location[2], location[1] + location[3]],
#                         [location[0], location[1] + location[3]]])
# print('rbox_in_img:', rbox_in_img)
#
# img_files = sorted(os.listdir(img_path))
# ims = [cv2.imread(os.path.join(img_path, imf)) for imf in img_files]
# ima = ims[f]
# location = rbox_in_img.flatten()
# cv2.polylines(ima, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
# cv2.imshow('bla', ima)
# cv2.waitKey(0)
