import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import pickle as pkl
from TrackerDyn import TrackerDyn
import matplotlib.pyplot as plt
from utils_vis_2 import *
from scipy.io import savemat
from utils_dynamics import *

# Tracker data
locs_gt = '/Users/marinaalonsopoal/Desktop/Objects/location_gt'
locs_pred = '/Users/marinaalonsopoal/Desktop/Objects/locations_SMOT_acrobats.obj'
scores = '/Users/marinaalonsopoal/Desktop/Objects/scores.obj'

with open(locs_gt, 'rb') as f:
    locs_gt = pkl.load(f)  # Manipulated on conv_cand_to_centr.py
with open(locs_pred, 'rb') as f:
    locs_pred = pkl.load(f)  # Directly from multiobject.py
with open(scores, 'rb') as f:
    scores = pkl.load(f)  # Directly from multiobject.py

# Parameters
T0 = 11  # System memory
eps = 1  # Noise variance
obj = 1
metric = 0  # if 0: JBLD, if 1: JKL
W = 3  # Smoothing window length


metric_name = ['JBLD', 'JKL']
scores = scores[obj]
loc_gt = locs_gt[obj]  # np.ndarray (154, 8)
loc = locs_pred[obj]  # list(154)
# locs_1[1] list(num_candidates for frame 1)
# locs_1[1][0][0]  np.ndarray (8,)

# NOTE: Al SiamMask anirà al revés: 1. Frame 2. Objecte, però en teoria no afecta

tracker_gt = TrackerDyn(T0=T0, noise=eps, metric=metric, not_GT=False)
tracker_pred = TrackerDyn(T0=T0, W=W, noise=eps, metric=metric)

for f in range(len(loc)):
    print('-----Frame:', f, '-----')
    # De moment treballo només amb el candidat cand, si vull canviar, iterar
    cand = 0
    loca = loc[f][cand][0]
    tracker_pred.update(loca)
    tracker_gt.update(loc_gt[f])

# Scores
scores_array = np.ones((len(scores)))
for f, s in enumerate(scores):
    scores_array[f] = -s


# Plotting
# plot_all_dist(tracker_pred, metric_name[metric], obj, T0, eps)

# plot_centr_and_dist(tracker_pred, tracker_gt, metric_name[metric], obj, T0, eps)

# NOTE: 8x3 plot
# plot_all(tracker_pred, tracker_gt, metric_name[metric], obj, T0, eps)
# # plot_locs_pred(tracker_gt, tracker_pred)

# NOTE: Plot Centroid and Velocity + JBLDS
# plot_centr_pos_and_vel(tracker_pred, tracker_gt, obj, metric_name[metric], T0, eps)

# NOTE: Normal vs. Smoothed
# plot_centr_and_jbld_2(tracker_pred, tracker_gt)

# NOTE: SLIDING vs. INCREASING
# plot_centr_and_jbld_3(tracker_pred, tracker_gt)

# NOTE: JBLD + Score
# plot_centr_and_jbld_4(tracker_pred, tracker_gt, scores_array)


#
# plot_centr_and_jbld(tracker_pred, tracker_gt, scores_array)

np.save('/Users/marinaalonsopoal/Desktop/Objects/centr_obj_1.npy', tracker_pred.buffer_centr)
# cx = tracker_pred.buffer_centr[:, 0]  # (154,)
# x = cx[45:56]  # Equivalent to MATLAB's x(45:55)
# T0 = 5
# x = cx[40:40+T0]
# # x = x - np.mean(x)
# x = x.reshape(len(x), 1)
# print('shape x', x.shape)
# H = Hankel(torch.from_numpy(x))
# print(H)
# print(H.shape)