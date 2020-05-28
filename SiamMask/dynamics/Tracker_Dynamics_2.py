from utils_dyn.utils_torch import *
import numpy as np


class TrackerDyn_2:

    def __init__(self, T0, R=5, W=1, t=0, noise=0.0001, metric=1, slow=False, norm=True):
        self.T0 = T0
        self.t = t
        self.noise = noise
        self.metric = metric
        self.W = W
        self.R = R
        self.slow = slow  # If true: Slow(but Precise), if false: Fast
        self.norm = norm  # If true: Norm, if false: MSE

        # Thresholds to base the classification decisions on
        self.th_jbld = 0.4  # 0.3
        self.th_jbld_max = 1
        self.th_eta = 25  # 15
        self.th_score = 0.97

        # Buffers with data
        self.buffer_pos = np.zeros([self.T0, 2])
        self.buffer_sz = np.zeros([self.T0, 2])

        # Buffers with distances and metrics
        self.dist_pos = np.zeros([self.T0, 2])  # cx, cy - Sliding Window
        self.dist_sz = np.zeros([self.T0, 2])  # w, h
        self.eta_centr = np.zeros([self.T0 - 1, 2])

        # Buffers with predicted
        # TODO: Aquests fan falta?
        self.pred_pos = np.zeros([1, 8])
        self.pred_centr = np.zeros([1, 8])

        # Buffers with flags
        self.predict_flag = np.zeros((self.T0, 1))
        self.buffer_pred_centr = np.zeros([self.T0, 2])

        self.xhat_list = [0]*(T0-1)

        # Buffer with scores
        self.scores = []
        self.scores_arr = np.asarray(self.scores)

    def update(self, pos, sz, sco):
        # print('self.t', self.t)
        self.scores.append(sco)
        self.scores_arr = np.asarray(self.scores)

        # if self.t > 0:  # Update velocity buffers
        #     self.buffer_centr_vel = np.vstack((self.buffer_centr_vel, c - self.buffer_centr[self.t - 1, :]))
        #     self.buffer_loc_vel = np.vstack((self.buffer_loc_vel, loc - self.buffer_loc[self.t - 1, :]))

        if self.t < self.T0:
            self.buffer_pos[self.t, :] = np.reshape(pos, (1, 2))
            self.buffer_sz[self.t, :] = np.reshape(sz, (1, 2))
        else:
            self.buffer_pos = np.vstack((self.buffer_pos,  np.reshape(pos, (1, 2))))
            self.buffer_sz = np.vstack((self.buffer_sz,  np.reshape(sz, (1, 2))))
            # c = self.classify()
            # if c:
            #     self.predict()
        self.t += 1
        pred = 1
        return pred

    def classify(self):
        # c: Boolean indicating if prediction must be done (True: Predict)
        c = False
        cond_jbld = self.dist_centr[self.t, :] >= self.th_jbld
        cond_jbld = cond_jbld.any()
        cond_eta = self.eta_centr[self.t, :] >= self.th_eta
        cond_eta = cond_eta.any()
        cond_score = self.scores_arr[self.t] <= self.th_score
        if cond_jbld and cond_eta and cond_score:
            c = True
        cond_jbld = self.dist_centr[self.t, :] >= self.th_jbld_max
        cond_jbld = cond_jbld.any()
        if cond_jbld:
            c = True
        # Update classification flag
        self.predict_flag = np.concatenate((self.predict_flag, np.asarray(c+1-1).reshape((1, 1))))
        return c

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

        data_pred = np.vstack((data_pred, pred_t))
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
            xhats = np.zeros((self.T0, 2))
            for d in range(2):
                data_root = self.buffer_centr[self.t - self.T0 + 1:self.t + 1, d]
                data_root = torch.from_numpy(data_root)
                data_root = data_root.view(1, len(data_root))
                data_root = data_root - torch.mean(data_root)
                [xhat, eta, mse] = fast_hstln_mo(data_root, self.R, self.slow)
                xhats[:, d] = xhat.numpy()

                if self.norm:
                    mses[0, d] = torch.norm(eta, 'fro').numpy()
                else:
                    mses[0, d] = mse.numpy()

            self.xhat_list.append(xhats)
            self.eta_centr = np.vstack((self.eta_centr, mses))

    def update_smooth(self):
        if self.W == 1:
            self.buffer_centr_smo = self.buffer_centr
        else:
            if self.t == 0:
                self.buffer_centr_smo = self.buffer_centr[0, :]
            self.buffer_centr_smo = self.smooth_data(self.buffer_centr, self.buffer_centr_smo)

    def smooth_data(self, data, smo_array):
        dim = data.shape[1]  # En el cas dels centroides, 2
        smo = np.zeros([1, dim])
        if self.t != 0:
            for d in range(dim):
                if self.t >= self.W:
                    seq = data[self.t - self.W:self.t, d]
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