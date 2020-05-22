import torch
import numpy as np
from utils_dynamics import *
device = torch.device('cpu')
from torch_utils import *


class TrackerDyn:

    def __init__(self, T0, R=5, W=1, t=0, noise=0.0001, metric=1, not_GT=True, slow=False, norm=True):
        self.T0 = T0
        self.t = t
        self.noise = noise
        self.metric = metric
        self.W = W
        self.not_GT = not_GT
        self.th = 0.5
        self.R = R
        self.slow = slow  # If true: Slow(but Precise), if false: Fast
        self.norm = norm  # If true: Norm, if false: MSE

        # Buffers with data
        self.buffer_loc = np.zeros([self.T0, 8])  # 4 corners position: [x1, y1, x2, y2 ...]
        self.buffer_loc_vel = np.zeros([1, 8])  # 4 corners velocity
        self.buffer_centr = np.zeros([self.T0, 2])  # 2 centroids position: [cx, cy]
        self.buffer_centr_vel = np.zeros([1, 2])  # 2 centroids velocity: [cx2-cx1, cy2-cy1]
        self.buffer_centr_vel_smo = np.zeros([1, 2])  # 2 centroids velocity: [cx2-cx1, cy2-cy1]

        # Buffers with distances and metrics
        self.dist_centr = np.zeros([self.T0, 2])  # cx, cy - Sliding Window
        self.eta_mse_centr = np.zeros([self.T0-1, 2])
        self.prediction = np.zeros([1, 8])

        # Buffers with flags
        self.predict_centr_flag = np.zeros([self.T0, 2])
        self.buffer_pred_centr = np.zeros([self.T0, 2])

        self.xhatt = 0


    def update(self, loc):
        # print('self.t', self.t)
        pred = 0
        c = compute_centroid(loc)

        if self.t > 0:  # Update velocity buffers
            self.buffer_centr_vel = np.vstack((self.buffer_centr_vel, c - self.buffer_centr[self.t-1, :]))
            self.buffer_loc_vel = np.vstack((self.buffer_loc_vel, loc - self.buffer_loc[self.t - 1, :]))

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
            self.update_dist_and_etas()  # Compute dynamic distances

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



    def update_dist_and_etas(self):
        if self.t >= self.T0:
            self.dist_centr = self.compute_dist(self.buffer_centr, self.dist_centr, 1, False)
        if self.t >= self.T0 - 1:
            mses = np.zeros((1, 2))
            for d in range(2):
                data_root = self.buffer_centr[self.t - self.T0 + 1:self.t + 1, d]
                data_root = torch.from_numpy(data_root)
                data_root = data_root.view(1, len(data_root))
                data_root = data_root - torch.mean(data_root)
                [xhat, eta, mse] = fast_hstln_mo(data_root, self.R, self.slow)
                if self.t == 36 and d==0:
                    self.xhatt = xhat.numpy()
                    print('xhaaaatt:', self.xhatt)

                if self.norm:
                    mses[0, d] = torch.norm(eta, 'fro').numpy()
                else:
                    mses[0, d] = mse.numpy()

            self.eta_mse_centr = np.vstack((self.eta_mse_centr, mses))




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


    def predict(self, data, preds):
        dim = data.shape[1]
        for d in range(8):
            data = self.buffer_loc[-(self.T0 + 1):-1, d]
            H = Hankel(data)
            preds[0, d] = predict_Hankel(H)
        self.prediction = preds
