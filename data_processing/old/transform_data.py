# This script transforms original data from tracker (4 corner locations) to smoothed data or velocity data

import numpy as np
import torch
import pickle as pkl
from TrackerDyn import TrackerDyn
import matplotlib.pyplot as plt
from utils_vis_2 import *

# Tracker data
locs_pred = '/Users/marinaalonsopoal/Desktop/Objects/locations_SMOT_acrobats.obj'

with open(locs_pred, 'rb') as f:
    locs_pred = pkl.load(f)  # Directly from multiobject.py

# Parameters
T0 = 5
eps = 0.0001
obj = 5
metric = 0  # if 0: JBLD, if 1: JKL

metric_name = ['JBLD', 'JKL']
loc = locs_pred[obj]  # list(154)
# locs_1[1] list(num_candidates for frame 1)
# locs_1[1][0][0]  np.ndarray (8,)

def smooth_data(window):
    """ Smoothes (computes mean) of a window given different candidates
    Args:
        - window: list containing different number of candidates per each time in the window
    Returns:
        - smoothed: list containing different smoothed candidates per one time
    """
    smoothed = []  # list that contains the smoothed values in the data format
    num_points = []  # list that contains the number of points for each time step
    [num_points.append(len(p)) for p in window]
    num_seqs = reduce(mul, num_points)  # number of combinations
    for i in range(num_seqs):
        seq = generate_seq_from_tree(num_points, window, i)
        x = torch.mean(seq[:, 0])
        y = torch.mean(seq[:, 1])
        m = np.asarray([x, y])
        smoothed.append([m])
    return smoothed


