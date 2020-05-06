from functools import reduce
from operator import mul
import torch
import numpy as np
from utils_dynamics import *
device = torch.device('cpu')


class TrackerDyn:

    def __init__(self, T0, t=0, noise=0.0001, metric=1):
        self.T0 = T0
        self.t = t
        self.noise = noise
        self.metric = metric
        self.buffer_loc = np.zeros([self.T0, 8])
        self.buffer_centr = np.zeros([self.T0, 2])
        self.dist_centr = np.zeros([self.T0, 2])  # cx, cy
        self.dist_centr_joint = np.zeros(self.T0)
        self.dist_loc = np.zeros([self.T0, 8])  # x1, y1, x2, ...
        self.dist_loc_joint = np.zeros(self.T0)
        self.prediction = np.zeros([1, 8])

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
        # if self.t == 55:
        #     self.predict()
        self.t += 1

    def dyn_dist(self):
        if self.t < self.T0 + 1:
            return 0  #TODO: Break?
        else:
            # Compute distances for centroids separately
            dist = np.zeros([1, 2])
            for d in range(2):
                data_root = self.buffer_centr[-(self.T0+1):-1, d]
                data = np.array([self.buffer_centr[-1, d]])
                dist[0, d] = compare_dyn(data_root, data, self.noise, self.metric)
            self.dist_centr = np.vstack((self.dist_centr, dist))

            # Compute distances for centroids jointly
            data_root = self.buffer_centr[-(self.T0 + 1):-1, :]
            data = self.buffer_centr[-1, :].reshape(1, 2)
            dist = np.array([compare_dyn(data_root, data, self.noise, self.metric)])
            self.dist_centr_joint = np.concatenate([self.dist_centr_joint, dist])

            # Compute distances for locations separately
            dist = np.zeros([1, 8])
            for d in range(8):
                data_root = self.buffer_loc[-(self.T0+1):-1, d]
                data = np.array([self.buffer_loc[-1, d]])
                dist[0, d] = compare_dyn(data_root, data, self.noise, self.metric)
            self.dist_loc = np.vstack((self.dist_loc, dist))

            # Compute distances for locations jointly
            data_root = self.buffer_loc[-(self.T0 + 1):-1, :]
            data = self.buffer_loc[-1, :].reshape(1, 8)
            dist = np.array([compare_dyn(data_root, data, self.noise, self.metric)])
            self.dist_loc_joint = np.concatenate([self.dist_loc_joint, dist])


    def predict(self):
        preds = np.zeros([1, 8])
        for d in range(8):
            data = self.buffer_loc[-(self.T0 + 1):-1, d]
            H = Hankel(data)
            preds[0, d] = predict_Hankel(H)
        self.prediction = preds