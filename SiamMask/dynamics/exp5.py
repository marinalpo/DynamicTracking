# exp5.py: Studies the results of Hankel prediction in a chosen circumstance

import torch
torch.set_default_dtype(torch.float64)
import pickle as pkl
from utils.utils_plots_dynamics import *
from utils.utils_dynamics import *
from TrackerDyn import TrackerDyn

# export PYTHONPATH=''/Users/marinaalonsopoal/PycharmProjects/DynamicTracking/''

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
slow = False  # If true: Slow(but Precise), if false: Fast
norm = True  # If true: Norm, if false: MSE

metric_name = ['JBLD', 'JKL']
scores = scores[obj]
loc_gt = locs_gt[obj]  # np.ndarray (154, 8)
loc = locs_pred[obj]  # list(154)
T = len(loc)
# locs_1[1] list(num_candidates for frame 1)
# locs_1[1][0][0]  np.ndarray (8,)
tin = 0
tfin = 60  #len(loc)

# NOTE: Al SiamMask anirà al revés: 1. Frame 2. Objecte, però en teoria no afecta

tracker_gt = TrackerDyn(T0=T0, R=R, noise=eps, metric=metric, not_GT=False)
tracker_pred = TrackerDyn(T0=T0, R=R,  W=W, noise=eps, metric=metric, slow=slow, norm=norm)
for f in range(tin, tfin):  # T
    print('-----Frame:', f, '-----')
    # De moment treballo només amb el candidat cand, si vull canviar, iterar
    cand = 0
    loca = loc[f][cand][0]
    tracker_pred.update(loca, scores[f])
    tracker_gt.update(loc_gt[f], scores[f])




d = 1  # d=0, x
t = 43  # Predecimos en este frame
xhats = tracker_pred.xhat_list
centr = tracker_pred.buffer_centr
xbad = centr[t, d]
print('xbad:', xbad)

# NOTE: Prediccion con x
x = centr[t-T0:t, d]
print('x:', x)
media = np.mean(x)
s_x = torch.from_numpy(x.reshape(x.shape[0], 1))
pred_x = predict_Hankel(Hankel(s_x))

print('pred x:', pred_x)

# NOTE: Prediccion con xhat
xhat= xhats[t-1]  # Sin la muestra corrupta
xhat_x = xhat[:, d]

xhat_x_med = xhat_x + media
print('xhat_x_med', xhat_x_med)
s_xhat = torch.from_numpy(xhat_x_med.reshape(xhat_x_med.shape[0], 1))
pred_xhat = predict_Hankel(Hankel(s_xhat))
print('pred xhat:', pred_xhat)

# # Plotting -------------------------------------------------------
fig, ax = plt.subplots(1, 1)
ax.set_title('Object 1 Prediction frames: ')
fr = np.arange(t-T0, t)
ax.scatter(fr, x, c='b', label='x', alpha=0.3)
ax.scatter(t, pred_x, c='b', edgecolor='k', label='pr. x')
ax.scatter(fr, xhat_x_med, c='r', label='xhat', alpha=0.3)
ax.scatter(t, pred_xhat, c='r', edgecolor='k', label='pr. xhat')
ax.scatter(t, xbad, c='y', edgecolor='k', label='bad')
ax.legend()
plt.show()



