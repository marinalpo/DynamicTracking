# from utils_dyn.utils_torch import *
import numpy as np
import torch
import statsmodels.api as sm
import pandas as pd
from sklearn import linear_model

# TODO: Else
from utils_dyn.utils_torch import *
from utils_dyn.utils_dynamics import *

# # TODO: When Debugging
# from SiamMask.utils_dyn.utils_plots_dynamics import *
# from SiamMask.utils_dyn.utils_dynamics import *

#
# from SiamMask.utils_dyn.utils_torch import *


class Dynamics_Module:

    def __init__(self, T0, t_init=1, eta_max_pred=10, both=True):

        # Parameters
        self.T0 = T0
        self.T0_ratio = 3
        self.t = 1
        self.t_init = t_init
        self.noise = 1
        self.metric = 1
        self.W = 1
        self.slow = False  # If true: Slow(but Precise), if false: Fast
        self.norm = True  # If true: Norm, if false: MSE
        self.eta_max_clas = 1
        self.eta_max_pred = eta_max_pred
        self.both = both

        # Thresholds to base the classification decisions on
        self.th_jbld = 0.5  # 0.3
        self.th_jbld_max = 0.75  # 1
        self.th_eta = 15  # 15
        self.th_score = 0.90  # 0.96
        self.th_score_min = 0.8  # 0.85

        # Buffers with data
        self.buffer_pos = np.zeros([self.T0, 2])
        self.buffer_pos_corr = np.zeros([self.T0, 2])  # CORRECTED
        self.buffer_pos_m = np.zeros([self.T0 - 1, 2])  # Means of the data
        self.buffer_pred = np.zeros([self.T0, 2])


        self.buffer_sz = np.zeros([self.T0, 2])
        self.buffer_ratio = np.zeros(self.T0)


        # Buffers with distances and metrics
        self.jbld_pos = np.zeros([self.T0, 2])  # cx, cy - JBLD with Sliding Window
        # self.dist_sz = np.zeros([self.T0, 2])  # w, h
        self.eta_norm_dif = np.zeros([self.T0, 2])
        self.Rs_clas = np.zeros([self.T0, 2])  # R: Dynamics returned by hstln when classifying
        self.Rs_pred = np.zeros([self.T0, 2])  # R: Dynamics returned by hstln when predicting

        # Buffers with flags
        self.predict_flag = [[False, False]] * self.T0
        self.buffer_pred_centr = np.zeros([self.T0, 2])
        self.xhat_list = [0] * (self.T0 - 1)

        # Buffer with scores
        self.scores = []
        self.scores_arr = np.asarray(self.scores)

    def update(self, pos, sz, sco):
        pred_pos = pos
        pred_ratio = sz[0]/sz[1]

        c = [False, False]

        # Fill buffers with incoming data
        self.scores.append(sco)
        self.scores_arr = np.asarray(self.scores)

        if self.t <= self.T0:
            self.buffer_pos[self.t - 1, :] = np.reshape(pos, (1, 2))
            self.buffer_pos_corr[self.t - 1, :] = np.reshape(pos, (1, 2))
            self.buffer_pred[self.t - 1, :] = np.reshape(pos, (1, 2))
            self.buffer_sz[self.t - 1, :] = np.reshape(sz, (1, 2))
            self.buffer_ratio[self.t-1] = sz[0]/sz[1]
        else:
            self.buffer_pos = np.vstack((self.buffer_pos, np.reshape(pos, (1, 2))))
            self.buffer_pos_corr = np.vstack((self.buffer_pos_corr, np.reshape(pos, (1, 2))))
            self.buffer_pred = np.vstack((self.buffer_pred, np.reshape(pos, (1, 2))))
            self.buffer_sz = np.vstack((self.buffer_sz, np.reshape(sz, (1, 2))))
            self.buffer_ratio = np.append(self.buffer_ratio, (sz[0]/sz[1]))

        # Compute distances, etas, classify and predict
        if self.t > self.T0:

            uhat = self.update_metrics()

            c = self.classify()

            if self.both:
                if c[0] or c[1]:
                    # print(' --- Predict both coordinates')
                    pred_pos[0] = self.predict(0)
                    # a = self.predict_hankel(0)
                    self.buffer_pos_corr[self.t - 1, 0] = pred_pos[0]
                    pred_pos[1] = self.predict(1)
                    # a = self.predict_hankel(1)
                    self.buffer_pos_corr[self.t - 1, 1] = pred_pos[1]
                    pred_ratio = self.predict_ratio()
                    self.buffer_ratio[self.t-1] = pred_ratio

                    self.buffer_pred[self.t - 1, 0] = pred_pos[0]
                    self.buffer_pred[self.t - 1, 1] = pred_pos[1]
            else:
                if c[0]:
                    print('Predict coordinate x')
                    pred_pos[0] = self.predict(0)
                    self.buffer_pos_corr[self.t - 1, 0] = pred_pos[0]
                    self.buffer_pred[self.t - 1, 0] = pred_pos[0]
                if c[1]:
                    print('Predict coordinate y')
                    pred_pos[1] = self.predict(1)
                    self.buffer_pos_corr[self.t - 1, 1] = pred_pos[1]
                    self.buffer_pred[self.t - 1, 1] = pred_pos[1]

        self.t += 1
        # print('Predicted:', pred_pos)
        return c, pred_pos, pred_ratio

    def predict_ratio(self):
        data = self.buffer_ratio[self.t - self.T0_ratio - 1:self.t - 1].reshape(-1, 1)
        # print('data to predict ratio:', data)
        x = np.arange(1, self.T0_ratio+1).reshape(-1, 1)
        regr = linear_model.LinearRegression()
        regr.fit(x, data)
        pred_ratio = regr.predict(np.asarray((self.T0_ratio+1)).reshape(-1, 1))
        # print('predicted ratio:', pred_ratio)
        return pred_ratio

    def update_bbox(self, pos, sz):
        x, y = pos
        self.buffer_pos_corr[-1, 0] = x
        self.buffer_pos_corr[-1, 1] = y
        ratio = sz[0]/sz[1]
        self.buffer_ratio[-1] = ratio

    def update_metrics(self):
        eta_dif = np.zeros((1, 2))
        jbld = np.zeros((1, 2))
        Rs = np.zeros((1, 2))
        Rs_pred = np.zeros((1, 2))

        for d in range(2):
            # print('d:', d)
            # data_root: Last T0 samples EXCLUDING the current one
            # TODO: Change buffer_pos_corr
            data_root = self.buffer_pos_corr[self.t - self.T0 - 1:self.t - 1, d]
            data_root = torch.from_numpy(data_root)
            data_root = data_root.view(1, len(data_root))
            # print('data root:', data_root)
            [uhat_root, eta_root, x, R] = fast_incremental_hstln_mo(data_root, self.eta_max_clas, self.slow)
            # print('norm eta:', torch.norm(eta_root, 'fro'))
            # print('uhat_root:', uhat_root)
            # print('R:', R)
            # u_hat = (u_hat.numpy()).flatten()
            # uhat[:, d] = u_hat
            Rs[0, d] = R
            # norm_eta[0, d] = torch.norm(eta, 'fro').numpy()

            # data_corr: Last T0 samples INCLUDING the current one
            # TODO: Change buffer_pos_corr
            data_corr = self.buffer_pos_corr[self.t - self.T0 - 1:self.t, d]
            data_corr = torch.from_numpy(data_corr)
            data_corr = data_corr.view(1, len(data_corr))
            [uhat_corr, eta_corr, x, mse] = fast_hstln_mo(data_corr, R, self.slow)

            eta_dif[0, d] = (torch.norm(eta_corr, 'fro') - torch.norm(eta_root, 'fro')).numpy()

            # Compute JBLD
            jbld[0, d] = compare_dyn_R(uhat_root.t(), uhat_corr.t(), self.noise, R, mean=False)
            # print('jbld:', jbld[0, d])

        self.jbld_pos = np.vstack((self.jbld_pos, jbld))
        self.Rs_clas = np.vstack((self.Rs_clas, Rs))
        self.Rs_pred = np.vstack((self.Rs_pred, Rs_pred))
        self.eta_norm_dif = np.vstack((self.eta_norm_dif, eta_dif))
        return uhat_root


    def classify(self):
        # c: Boolean indicating if prediction must be done (True: Predict)
        c = [False, False]
        for d in range(2):
            cond_jbld = self.jbld_pos[self.t - 1, d] >= self.th_jbld
            cond_eta = self.eta_norm_dif[self.t - 1, d] >= self.th_eta
            cond_score = self.scores_arr[self.t - 1] <= self.th_score
            if cond_jbld and cond_eta and cond_score:
                c[d] = True
            cond_jbld = self.jbld_pos[self.t - 1, d] >= self.th_jbld_max
            if cond_jbld:
                c[d] = True
            cond_score = self.scores_arr[self.t - 1] <= self.th_score_min
            if cond_score:
                c[d] = True
            # cond_R = self.Rs_clas[self.t-1, d] - self.Rs_clas[self.t-2, d] != 0
            # if cond_R and self.t != self.T0 + 1:
            #     c[d] = True

        # Update classification flag
        # Si es vol canviar a que es prediguin els dos fer algo tipo .any()
        self.predict_flag.append(c)
        return c

    def predict(self, coord):
        # print('predict ssest')
        # xhat: (self.T0, 2)
        # flag: (self.t - 1, dim)
        # Return pred: (self.t, dim)
        # pred = np.zeros((1, 2))
        d = coord
        # TODO: POSSAR EL self.buffer_pos_corr
        data_root = self.buffer_pos_corr[self.t - self.T0 - 1:self.t - 1, d]
        u_ts = data_root
        # print('u_ts shape:', u_ts.shape)
        data_root = torch.from_numpy(data_root)
        data_root = data_root.view(1, len(data_root))
        [uhat_root, eta_root, x, R] = fast_incremental_hstln_mo(data_root, self.eta_max_pred, self.slow)
        self.Rs_pred[self.t - 1, d] = R

        # 1. Construct a model instance with our dataset

        # uhat = uhat_root.numpy()
        # uhat = uhat.squeeze()
        # u_ts = uhat
        # u_ts = u_ts.astype(int)
        # print('type uts 0', type(u_ts[0]))

        # print('before prediction:', u_ts)
        ts = pd.DataFrame(u_ts)
        ts.columns = ['u']


        # Step 1: construct an SARIMAX model for US inflation data
        # https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX
        model = sm.tsa.SARIMAX(ts.u, order=(R, 0, 0), trend='ct', measurement_error=True)

        # Step 2: fit the model's parameters by maximum likelihood
        results = model.fit(disp=0, maxiter=200, method='nm')

        # Step 3: forecast results
        pred = results.forecast(1)
        # pred = results.forecast(1).to_numpy()

        # print('Prediction:', pred)
        # print('prediction ssest:', pred)


        return pred


    def predict_hankel(self, coord):
        print('predict amb mario')
        # xhat: (self.T0, 2)
        # flag: (self.t - 1, dim)
        # Return pred: (self.t, dim)
        # pred = np.zeros((1, 2))

        d = coord
        # TODO: POSSAR EL self.buffer_pos_corr
        data_root = self.buffer_pos_corr[self.t - self.T0 - 1:self.t - 1, d]
        m = np.mean(data_root)

        # print('u_ts shape:', u_ts.shape)
        data_root = torch.from_numpy(data_root)
        data_root = data_root.view(1, len(data_root))
        [uhat_root, eta_root, x, R] = fast_incremental_hstln_mo(data_root, self.eta_max_pred, self.slow)
        self.Rs_pred[self.t - 1, d] = R
        u_ts = uhat_root.t()
        nr = R + 1
        nc = self.T0 - R
        # print('u_ts.shape', u_ts.shape)
        H = Hankel_new(u_ts, nr, nc, mean=False)
        pred = predict_Hankel(H)
        # print('prediction mario:', pred)

        return pred


