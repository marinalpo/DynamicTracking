import numpy as np
import torch
import pickle as pkl
from TrackerDyn import TrackerDyn
import matplotlib.pyplot as plt
from utils_vis_2 import *

# Tracker data
locs_gt = '/Users/marinaalonsopoal/Desktop/location_gt'
locs_pred = '/Users/marinaalonsopoal/Desktop/locations_SMOT_acrobats.obj'

with open(locs_gt, 'rb') as f:
    locs_gt = pkl.load(f)  # Manipulated on conv_cand_to_centr.py
with open(locs_pred, 'rb') as f:
    locs_pred = pkl.load(f)  # Directly from multiobject.py

# Parameters
T0 = 4
eps = 0.0001
obj = 5
metric = 0  # if 0: JBLD, if 1: JKL

metric_name = ['JBLD', 'JKL']
loc_gt = locs_gt[obj]  # np.ndarray (154, 8)
loc = locs_pred[obj]  # list(154)
# locs_1[1] list(num_candidates for frame 1)
# locs_1[1][0][0]  np.ndarray (8,)

# NOTE: Al SiamMask anirà al revés: 1. Frame 2. Objecte, però en teoria no afecta

tracker_gt = TrackerDyn(T0=T0, noise=eps, metric=metric)
tracker_pred = TrackerDyn(T0=T0, noise=eps, metric=metric)

jblds_pred = np.zeros(len(loc))

for f in range(len(loc)):
    print('-----Frame:', f, '-----')
    # De moment treballo només amb el candidat cand, si vull canviar, iterar
    cand = 0
    loca = loc[f][cand][0]
    tracker_pred.update(loca)
    tracker_pred.dyn_dist()
    tracker_gt.update(loc_gt[f])

# Plotting
# plot_all_dist(tracker_pred, metric_name[metric], obj, T0, eps)
# plot_centr_and_dist(tracker_pred, tracker_gt, metric_name[metric], obj, T0, eps)
plot_all(tracker_pred, tracker_gt, metric_name[metric], obj, T0, eps)
# plot_locs_pred(tracker_gt, tracker_pred)
# # Find maximum of joint corners
# a = np.argmax(tracker_pred.dist_loc_joint)
# print(a)