import torch
torch.set_default_dtype(torch.float64)
import pickle as pkl
from utils_dyn.utils_plots_dynamics import *
from Tracker_Dynamics import TrackerDyn

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
tfin = len(loc)

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


plot_jbld_eta_score(tracker_pred, tracker_gt, obj, norm, slow, tin, tfin)


