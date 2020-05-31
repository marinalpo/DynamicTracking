# from utils_dyn.utils_torch import *
import numpy as np
import torch

from utils_dyn.utils_torch import *

#
# from SiamMask.utils_dyn.utils_torch import *


class TrackerDyn_2:

    def __init__(self, T0, R=5, W=1, t=1, noise=0.0001, metric=1, slow=False, norm=True):

        # Parameters
        self.T0 = T0
        self.t = t
        self.noise = noise
        self.metric = metric
        self.W = W
        self.R = R
        self.slow = slow  # If true: Slow(but Precise), if false: Fast
        self.norm = norm  # If true: Norm, if false: MSE

        # Thresholds to base the classification decisions on
        self.th_jbld = 0.5  # 0.3
        self.th_jbld_max = 1  # 1
        self.th_eta = 15  # 15
        self.th_score = 0.97
        self.th_score_min = 0.9

        # Buffers with data
        self.buffer_pos = np.zeros([self.T0, 2])
        self.buffer_pos_corr = np.zeros([self.T0, 2])  # CORRECTED
        self.buffer_sz = np.zeros([self.T0, 2])
        self.buffer_pos_m = np.zeros([self.T0 - 1, 2])  # Means of the data

        # Buffers with distances and metrics
        self.dist_pos = np.zeros([self.T0, 2])  # cx, cy - Sliding Window
        # self.dist_sz = np.zeros([self.T0, 2])  # w, h
        self.eta_pos = np.zeros([self.T0 - 1, 2])

        # Buffers with flags
        self.predict_flag = [[False, False]] * self.T0
        self.buffer_pred_centr = np.zeros([self.T0, 2])

        self.xhat_list = [0] * (self.T0 - 1)

        # Buffer with scores
        self.scores = []
        self.scores_arr = np.asarray(self.scores)

    def update(self, pos, sz, sco):
        pred_pos = pos

        c = [False, False]

        # Fill buffers with incoming data
        self.scores.append(sco)
        self.scores_arr = np.asarray(self.scores)
        if self.t <= self.T0:
            self.buffer_pos[self.t - 1, :] = np.reshape(pos, (1, 2))
            self.buffer_pos_corr[self.t - 1, :] = np.reshape(pos, (1, 2))
            self.buffer_sz[self.t - 1, :] = np.reshape(sz, (1, 2))
        else:
            self.buffer_pos = np.vstack((self.buffer_pos, np.reshape(pos, (1, 2))))
            self.buffer_pos_corr = np.vstack((self.buffer_pos_corr, np.reshape(pos, (1, 2))))
            self.buffer_sz = np.vstack((self.buffer_sz, np.reshape(sz, (1, 2))))

        # Compute distances, etas, classify and predict
        if self.t >= self.T0:
            self.update_etas()

            if self.t != self.T0:
                self.dist_pos = self.compute_dist(self.buffer_pos_corr, self.dist_pos, 1, False)
                c = self.classify()
                if c[0] or c[1]:
                    if c[0]:
                        print('Lets predict x!')
                        pred_pos[0] = self.predict(0)
                        self.buffer_pos_corr[self.t - 1, 0] = pred_pos[0]
                    if c[1]:
                        print('Lets predict y!')
                        pred_pos[1] = self.predict(1)
                        self.buffer_pos_corr[self.t - 1, 1] = pred_pos[1]

        self.t += 1
        return c, pred_pos

    def classify(self):
        # c: Boolean indicating if prediction must be done (True: Predict)
        c = [False, False]
        for d in range(2):
            cond_jbld = self.dist_pos[self.t - 1, d] >= self.th_jbld
            cond_eta = self.eta_pos[self.t - 1, d] >= self.th_eta
            cond_score = self.scores_arr[self.t - 1] <= self.th_score
            if cond_jbld and cond_eta and cond_score:
                c[d] = True
            cond_jbld = self.dist_pos[self.t - 1, d] >= self.th_jbld_max
            if cond_jbld:
                c[d] = True
            cond_score = self.scores_arr[self.t - 1] <= self.th_score_min
            if cond_score:
                c[d] = True

        # Update classification flag
        # TODO: Si es vol canviar a que es prediguin els dos fer algo tipo .any()
        self.predict_flag.append(c)
        return c

    def predict(self, coord):
        # xhat: (self.T0, 2)
        # flag: (self.t - 1, dim)
        # Return pred: (self.t, dim)
        # pred = np.zeros((1, 2))
        predict_type = 3  # Predict with data not xhat
        if predict_type == 1:
            for d in range(2):
                # TODO: POSSAR EL self.buffer_pos_corr
                data = self.buffer_pos_corr[self.t - self.T0 - 1: self.t - 1, d]
                # data = self.buffer_pos[self.t - self.T0 - 1: self.t - 1, d]
                data = torch.from_numpy(data.reshape(self.T0, 1))
                print('data:', data)
                H = Hankel(data)
                pred[0, d] = predict_Hankel(H)
                print('pred:', pred[0, d])

        elif predict_type == 2:
            xhat = self.xhat_list[self.t - 2]
            for d in range(2):
                data = xhat[:, d] + self.buffer_pos_m[self.t - 2, d]
                data = data.reshape(self.T0, 1)
                data = torch.from_numpy(data)
                print('data:', data)
                H = Hankel(data)
                pred[0, d] = predict_Hankel(H)
                print('pred:', pred[0, d])
        else:
            for d in range(2):
                d = coord
                # TODO: POSSAR EL self.buffer_pos_corr
                data = self.buffer_pos[self.t - self.T0 - 1: self.t - 1, d]
                diff = data[-1] - data[-2]
                pred = data[-1] + diff

        return pred

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
                    data_root = buffer[self.t - self.T0 - 1:self.t - 1, d].reshape(self.T0, 1)
                elif kind == 2:  # INCREASING Window
                    data_root = buffer[0:self.t, d].reshape(self.t, 1)
                data = np.array([buffer[self.t - 1, d]]).reshape(1, 1)
                dist[0, d] = compare_dyn(data_root, data, self.noise)
            dist_array = np.vstack((dist_array, dist))
        return dist_array

    def update_etas(self):
        mses = np.zeros((1, 2))
        xhats = np.zeros((self.T0, 2))
        means = np.zeros((1, 2))
        for d in range(2):
            data_root = self.buffer_pos_corr[self.t - self.T0:self.t, d]
            data_root = torch.from_numpy(data_root)
            data_root = data_root.view(1, len(data_root))
            means[0, d] = torch.mean(data_root)
            data_root = data_root - means[0, d]
            [xhat, eta, mse] = fast_hstln_mo(data_root, self.R, self.slow)
            xhats[:, d] = xhat.numpy()

            if self.norm:
                mses[0, d] = torch.norm(eta, 'fro').numpy()
            else:
                mses[0, d] = mse.numpy()

        self.xhat_list.append(xhats)
        self.eta_pos = np.vstack((self.eta_pos, mses))
        self.buffer_pos_m = np.vstack((self.buffer_pos_m, means))
