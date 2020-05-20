from functools import reduce
from operator import mul
import torch
import numpy as np
from utils_dynamics import *
device = torch.device('cpu')


class TrackerDyn:

    def __init__(self, T0, W=1, t=0, noise=0.0001, metric=1, not_GT=True):
        self.T0 = T0
        self.t = t
        self.noise = noise
        self.metric = metric
        self.W = W
        self.not_GT = not_GT
        self.th = 0.5

        # Buffers with data
        self.buffer_loc = np.zeros([self.T0, 8])  # 4 corners position: [x1, y1, x2, y2 ...]
        self.buffer_loc_vel = np.zeros([1, 8])  # 4 corners velocity
        self.buffer_centr = np.zeros([self.T0, 2])  # 2 centroids position: [cx, cy]
        self.buffer_centr_vel = np.zeros([1, 2])  # 2 centroids velocity: [cx2-cx1, cy2-cy1]
        self.buffer_centr_vel_smo = np.zeros([1, 2])  # 2 centroids velocity: [cx2-cx1, cy2-cy1]
        self.buffer_centr_smo = np.zeros([1, 2])

        # Buffers with distances
        self.dist_centr = np.zeros([self.T0, 2])  # cx, cy - Sliding Window
        self.dist_centr_smo = np.zeros([self.T0, 2])  # cx, cy - Sliding Window
        self.dist_centr_smo_2 = np.zeros([self.T0, 2])  # cx, cy - Increasing Window
        self.dist_centr_2 = np.zeros([self.T0, 2])  # cx, cy - Increasing Window
        self.dist_centr_vel = np.zeros([self.T0, 2])  # cx_vel, cy_vel
        self.dist_centr_vel_smo = np.zeros([self.T0, 2])  # cx_vel, cy_vel
        self.dist_centr_vel_2 = np.zeros([self.T0, 2])  # cx_vel, cy_vel
        self.dist_centr_joint = np.zeros([self.T0, 1])
        self.dist_centr_joint_2 = np.zeros([self.T0, 1])
        self.dist_loc = np.zeros([self.T0, 8])  # x1, y1, x2, ...
        self.dist_loc_joint = np.zeros([self.T0, 1])
        self.prediction = np.zeros([1, 8])

        # Buffers with flags
        self.predict_centr_flag = np.zeros([self.T0, 2])
        self.buffer_pred_centr = np.zeros([self.T0, 2])


    def update(self, loc):
        # print('self.t', self.t)
        pred = 0
        c = compute_centroid(loc)


        if self.t > 0:  # Update velocity buffers
            self.buffer_centr_vel = np.vstack((self.buffer_centr_vel, c - self.buffer_centr[self.t-1, :]))
            self.buffer_loc_vel = np.vstack((self.buffer_loc_vel, loc - self.buffer_loc[self.t - 1, :]))

        # if self.t <
        if self.t < self.T0:
            self.buffer_loc[self.t, :] = loc
            self.buffer_centr[self.t, :] = c
        else:
            loc = np.reshape(loc, (-1, 8))
            c = np.reshape(c, (-1, 2))
            self.buffer_loc = np.vstack((self.buffer_loc, loc))
            self.buffer_centr = np.vstack((self.buffer_centr, c))

        if self.not_GT:
            self.update_smooth()
            if self.t > 0:  # Update velocity buffers
                self.buffer_centr_vel_smo = np.vstack((self.buffer_centr_vel_smo, self.buffer_centr_smo[self.t, :] -
                                                       self.buffer_centr_smo[self.t - 1, :]))

            self.update_dist()  # Compute dynamic distances
        # Classify into predict or not predict
        # if self.t >= self.T0:
        #
        #     self.predict_centr_flag = self.classify_and_predict(self.dist_centr_smo, self.buffer_centr_smo,
        #                                                         self.predict_centr_flag, self.buffer_pred_centr)
        # elif self.t == self.T0 - 1:
        #     self.buffer_pred_centr = self.buffer_centr_smo
        #     print('')


        self.t += 1
        return pred


    def classify_and_predict(self, data_dist, data_centr, flag, data_pred):

        # data: (self.t, dim)
        # flag: (self.t - 1, dim)
        # Return flag: (self.t, dim)
        dim = data_dist.shape[1]
        flag_t = np.zeros((1, dim))
        pred_t = np.zeros((1, dim))
        for d in range(dim):
            if data_dist[self.t, d] >= self.th:
                print('We are gonna predict bitches!!!!')
                flag_t[0, d] = 1
                data_root = data_centr[self.t - self.T0:self.t, d].reshape(self.T0, 1)
                print('len data root:', data_root.shape)
                H = Hankel(data_root)
                pred_t[0, d] = predict_Hankel(H)
            else:
                pred_t[0, d] = data_centr[self.t, d]

        data_pred =np.vstack((data_pred, pred_t))
        flag = np.vstack((flag, flag_t))
        return flag


    def compute_dist(self, buffer, dist_array, kind, joint):
        dim = buffer.shape[1]
        if joint:
            if kind == 1:  # SLIDING Window
                data_root = buffer[self.t - self.T0:self.t, :].reshape(self.T0, dim)
            elif kind == 2:  # INCREASING Window
                data_root = buffer[0:self.t, :].reshape(self.t, dim)
            data = np.array([buffer[self.t, :]]).reshape(1, dim)
            dist = compare_dyn(data_root, data, self.noise)
            dist_array = np.vstack((dist_array, dist))
        else:
            dist = np.zeros([1, dim])
            for d in range(dim):
                if kind == 1:  # SLIDING Window
                    data_root = buffer[self.t - self.T0:self.t, d].reshape(self.T0, 1)
                elif kind == 2:  # INCREASING Window
                    data_root = buffer[0:self.t, d].reshape(self.t, 1)
                data = np.array([buffer[self.t, d]]).reshape(1, 1)
                dist[0, d] = compare_dyn(data_root, data, self.noise)
            dist_array = np.vstack((dist_array, dist))
        return dist_array



    def update_dist(self):
        if self.t >= self.T0:
            self.dist_centr = self.compute_dist(self.buffer_centr, self.dist_centr, 1, False)
            self.dist_centr_smo = self.compute_dist(self.buffer_centr_smo, self.dist_centr_smo, 1, False)
            self.dist_centr_smo_2 = self.compute_dist(self.buffer_centr_smo, self.dist_centr_smo_2, 2, False)


            self.dist_centr_2 = self.compute_dist(self.buffer_centr, self.dist_centr_2, 2, False)
            self.dist_centr_vel = self.compute_dist(self.buffer_centr_vel, self.dist_centr_vel, 1, False)
            self.dist_centr_vel_smo = self.compute_dist(self.buffer_centr_vel_smo, self.dist_centr_vel_smo, 1, False)
            self.dist_centr_vel_2 = self.compute_dist(self.buffer_centr_vel, self.dist_centr_vel_2, 2, False)
            self.dist_centr_joint = self.compute_dist(self.buffer_centr, self.dist_centr_joint, 1, True)
            self.dist_centr_joint_2 = self.compute_dist(self.buffer_centr, self.dist_centr_joint_2, 2, True)

            self.dist_loc = self.compute_dist(self.buffer_loc, self.dist_loc, 1, False)
            self.dist_loc_joint = self.compute_dist(self.buffer_loc, self.dist_loc_joint, 1, True)

    def update_smooth(self):
        if self.W == 1:
            self.buffer_centr_smo = self.buffer_centr
        else:
            if self.t == 0:
                self.buffer_centr_smo = self.buffer_centr[0,:]
            self.buffer_centr_smo = self.smooth_data(self.buffer_centr, self.buffer_centr_smo)



    def smooth_data(self, data, smo_array):
        dim = data.shape[1]  # En el cas dels centroides, 2
        smo = np.zeros([1, dim])
        if self.t != 0:
            for d in range(dim):
                if self.t >= self.W:
                    seq = data[self.t-self.W:self.t, d]
                else:
                    seq = data[0:self.t, d]
                smo[0, d] = np.mean(seq)
            smo_array = np.vstack((smo_array, smo))
        return smo_array




    def predict(self):
        dim = data.shape[1]
        for d in range(8):
            data = self.buffer_loc[-(self.T0 + 1):-1, d]
            H = Hankel(data)
            preds[0, d] = predict_Hankel(H)
        self.prediction = preds


