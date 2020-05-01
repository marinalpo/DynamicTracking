from functools import reduce
from operator import mul
import torch
import numpy as np
from utils_dynamics import *
device = torch.device('cpu')

def compare_dynamics(data_root, data, eps, coord, BS=1):
    """ Compares dynamics between two sequences
    Args:
        - data_root: Data sequence
        - data: Data sequence including data_root
        - eps: Noise value
        - coord: Coordinate value (0: x, 1: y, 2: mean)
        - BS: Batch Size
    Returns:
        - dist: (BS, 2) JBLD distance between sequences
    """
    dist = torch.zeros(2, device=device)
    for d in range(2):
        H0 = Hankel(data_root[:, d])
        H1 = Hankel(data_root[:, d], True, data[:, d])
        dist[d] = JBLD(Gram(H0, eps), Gram(H1, eps), True)
    if coord == 2:
        dist = torch.mean(dist)
    else:
        dist = dist[coord]  # Print only x information
    return dist


class TrackerDyn:

    def __init__(self, T0, t=0, noise=0.0001, metric=1):
        self.T0 = T0
        self.t = t
        self.noise = noise
        self.metric = metric
        self.buffer_loc = np.zeros([T0, 8])
        self.buffer_centr = np.zeros([T0, 2])
        self.JBLDs = np.zeros(T0)

    def update(self, loc):
        c = compute_centroid(loc)
        if self.t < self.T0:
            self.buffer_loc[self.t, :] = loc
            self.buffer_centr[self.t, :] = c
        else:
            loc = np.reshape(loc, (-1, 8))
            c = np.reshape(c, (-1, 2))
            self.buffer_loc = np.vstack((self.buffer_loc, loc))
            self.buffer_centr = np.vstack((self.buffer_centr, c))
        self.t += 1

    def jbld(self):
        if self.t < self.T0 + 1:
            return 0
        else:
            data_root = self.buffer_centr[-(self.T0+1):-1, :]
            data = self.buffer_centr[-1, :].reshape(1, 2)
            # print('data_root', data_root)
            # print('data', data)
            jbld = compare_dyn(data_root, data, self.noise, self.metric)
        return jbld.numpy()

    # def predict(self):
    #     Hx = Hankel(self.buffer_past_x)
    #     Hy = Hankel(self.buffer_past_y)
    #     px = predict_Hankel(Hx)
    #     py = predict_Hankel(Hy)