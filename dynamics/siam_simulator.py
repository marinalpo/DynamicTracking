import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import pickle as pkl
from TrackerDyn import TrackerDyn
import matplotlib.pyplot as plt
from utils_vis_2 import *
from scipy.io import savemat
from utils_dynamics import *

# Tracker data
locs_gt = '/Users/marinaalonsopoal/Desktop/Objects/location_gt'
locs_pred = '/Users/marinaalonsopoal/Desktop/Objects/locations_SMOT_acrobats.obj'
scores = '/Users/marinaalonsopoal/Desktop/Objects/scores.obj'

with open(locs_gt, 'rb') as f:
    locs_gt = pkl.load(f)  # Manipulated on conv_cand_to_centr.py
with open(locs_pred, 'rb') as f:
    locs_pred = pkl.load(f)  # Directly from multiobject.py
with open(scores, 'rb') as f:
    scores = pkl.load(f)  # Directly from multiobject.py

# Parameters
T0 = 11  # System memory
R = 5
eps = 1  # Noise variance
obj = 2
metric = 0  # if 0: JBLD, if 1: JKL
W = 3  # Smoothing window length
slow = True  # If true: Slow(but Precise), if false: Fast
norm = True  # If true: Norm, if false: MSE


metric_name = ['JBLD', 'JKL']
scores = scores[obj]
loc_gt = locs_gt[obj]  # np.ndarray (154, 8)
loc = locs_pred[obj]  # list(154)
T = len(loc)
# locs_1[1] list(num_candidates for frame 1)
# locs_1[1][0][0]  np.ndarray (8,)

# NOTE: Al SiamMask anirà al revés: 1. Frame 2. Objecte, però en teoria no afecta

tracker_gt = TrackerDyn(T0=T0, R=R, noise=eps, metric=metric, not_GT=False)
tracker_pred = TrackerDyn(T0=T0, R=R,  W=W, noise=eps, metric=metric, slow=slow, norm=norm)
tin = 0
tfin = 37
for f in range(tin, tfin):  # T
    print('-----Frame:', f, '-----')
    # De moment treballo només amb el candidat cand, si vull canviar, iterar
    cand = 0
    loca = loc[f][cand][0]
    tracker_pred.update(loca)
    tracker_gt.update(loc_gt[f])



t = 36
x_bef_1 = tracker_pred.buffer_centr
x_bef = x_bef_1[t - T0+1:t+1, 0]
x_bef_10 = x_bef_1[t - T0+1:t, 0]
m = np.mean(x_bef)
xhat = tracker_pred.xhatt
xhat_10 = xhat[:, 0:-1]
print('xhat.shape', xhat.shape)
xhat_m = xhat + m
xhat_m_10 = xhat_10+m


print('len xhat:', len(xhat))

s1 = torch.from_numpy(x_bef_10.reshape(len(x_bef_10), 1))
H1 = Hankel(s1)
x1 = predict_Hankel(H1)
print('x1', x1)

s2 = torch.from_numpy(xhat_m_10.reshape(xhat_10.shape[1], 1))
H2 = Hankel(s2)
x2 = predict_Hankel(H2)
print('x2', x2)



fig, ax = plt.subplots(1, 1)
ax.set_title('Object 2 Prediction frames: 25-36')
fr = np.arange(len(x_bef))
ax.scatter(fr, x_bef, c='b', label='x', alpha=0.3)
ax.scatter(10, x1, c='b', edgecolor='k', label='pr. x')
ax.scatter(fr, xhat_m, c='r', label='xhat', alpha=0.3)
ax.scatter(10, x2, c='r', edgecolor='k', label='pr. xhat')
ax.legend()
plt.show()


# Scores
scores_array = np.ones((len(scores)))
for f, s in enumerate(scores):
    scores_array[f] = -s

scores_array = scores_array[tin:tfin]

# Plotting -------------------------------------------------------
slow_name = ['Fast', 'Slow']
norm_name = ['MSE', 'NORM']

frames = np.arange(tin, tfin)
centr_gt = tracker_gt.buffer_centr
centr_pred = tracker_pred.buffer_centr
etas = tracker_pred.eta_mse_centr
jbld = tracker_pred.dist_centr

max_jbld = np.max(jbld)
max_etas = np.max(etas)

s_gt = 25
s_pred = 25
s_p = 5

eps = tracker_pred.noise
T0 = tracker_pred.T0
R = tracker_pred.R

fig, ax = plt.subplots(4, 2)
fig.tight_layout()

for i in range(4):
    for j in range(2):
        ax[i, j].grid(axis='x', zorder=1, alpha=0.4)

fig.suptitle('Object:' + str(obj))

ax[0, 0].set_title('Position Centroid x')
ax[0, 0].scatter(frames, centr_pred[:, 0], c='r', s=s_pred, edgecolors='k', alpha=1, label='Predictions', zorder=3)
ax[0, 0].scatter(frames, centr_gt[:, 0], c='r', s=s_gt, alpha=0.2, label='GT', zorder=3)
ax[0, 0].legend()

ax[0, 1].set_title('Position Centroid y')
ax[0, 1].scatter(frames, centr_pred[:, 1], c='g', s=s_pred, edgecolors='k', alpha=1, label='Predictions', zorder=3)
ax[0, 1].scatter(frames, centr_gt[:, 1], c='g', s=s_gt, alpha=0.2, label='GT', zorder=3)
ax[0, 1].legend()

ax[1, 0].plot(frames, jbld[:, 0], c='r')
ax[1, 0].set_title('JBLD distance in Centroid x using T0:' + str(T0) + ' and Noise:' + str(eps))
ax[1, 0].set_ylim([0, max_jbld + max_jbld * 0.05])

ax[1, 1].plot(frames, jbld[:, 1], c='g')
ax[1, 1].set_title('JBLD distance in Centroid y using T0:' + str(T0) + ' and Noise:' + str(eps))
ax[1, 1].set_ylim([0, max_jbld + max_jbld * 0.05])


ax[2, 0].plot(frames, etas[:, 0], c='r')
ax[2, 0].set_title(slow_name[slow] + ' Implementation of ' + norm_name[norm] + ' of Eta in Centroid x '
                    'using T0:' + str(T0) + ' and R:' + str(R))
ax[2, 0].set_ylim([0, max_etas + max_etas * 0.05])

ax[2, 1].plot(frames, etas[:, 1], c='g')
ax[2, 1].set_title(slow_name[slow] + ' Implementation of ' + norm_name[norm] + ' of Eta in Centroid y '
                    'using T0:' + str(T0) + ' and R:' + str(R))
ax[2, 1].set_ylim([0, max_etas + max_etas * 0.05])

ax[3, 0].plot(frames, scores_array, c='k')
ax[3, 0].set_title('Appearance-based tracker negative score (confidence)')
ax[3, 1].plot(frames, scores_array, c='k')
ax[3, 1].set_title('Appearance-based tracker negative score (confidence)')
ax[3, 0].set_xlabel('frame')
ax[3, 1].set_xlabel('frame')

plt.show()


