import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import pickle as pkl
from TrackerDyn import TrackerDyn
import matplotlib.pyplot as plt
from utils_vis_2 import *
from scipy.io import savemat

# Tracker data
locs_gt = '/Users/marinaalonsopoal/Desktop/Objects/location_gt'
locs_pred = '/Users/marinaalonsopoal/Desktop/Objects/locations_SMOT_acrobats.obj'

with open(locs_gt, 'rb') as f:
    locs_gt = pkl.load(f)  # Manipulated on conv_cand_to_centr.py
with open(locs_pred, 'rb') as f:
    locs_pred = pkl.load(f)  # Directly from multiobject.py

# Parameters
T0 = 5  # System memory
eps = 2  # Noise variance
obj = 5
metric = 0  # if 0: JBLD, if 1: JKL
W = 5  # Smoothing window length

metric_name = ['JBLD', 'JKL']
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

# Plotting
# plot_all_dist(tracker_pred, metric_name[metric], obj, T0, eps)

# plot_centr_and_dist(tracker_pred, tracker_gt, metric_name[metric], obj, T0, eps)

# NOTE: 8x3 plot
# plot_all(tracker_pred, tracker_gt, metric_name[metric], obj, T0, eps)
# plot_locs_pred(tracker_gt, tracker_pred)

# NOTE: Plot Centroid and Velocity + JBLDS
plot_centr_pos_and_vel(tracker_pred, tracker_gt, obj, metric_name[metric], T0, eps)


# plot_centr_and_jbld(tracker_pred, tracker_gt)  # NOTE: Normal vs. Smoothed
# plot_centr_and_jbld_3(tracker_pred, tracker_gt)  # NOTE: SLIDING vs. INCREASING
