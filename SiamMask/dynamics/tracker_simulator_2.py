import torch
torch.set_default_dtype(torch.float64)
import pickle as pkl
from utils_dyn.utils_plots_dynamics import *
from Tracker_Dynamics_2 import TrackerDyn_2


# Tracker data
target_sz_dict = '/Users/marinaalonsopoal/Desktop/Objects/positions/target_sz_dict.obj'
target_pos_dict = '/Users/marinaalonsopoal/Desktop/Objects/positions/target_pos_dict.obj'
locs_gt = '/Users/marinaalonsopoal/Desktop/Objects/location_gt'
scores = '/Users/marinaalonsopoal/Desktop/Objects/scores.obj'

with open(locs_gt, 'rb') as f:
    locs_gt = pkl.load(f)  # Manipulated on conv_cand_to_centr.py
with open(target_sz_dict, 'rb') as f:
    target_sz_dict = pkl.load(f)  # Directly from multi_pred.py
with open(target_pos_dict, 'rb') as f:
    target_pos_dict = pkl.load(f)  # Directly from multi_pred.py
with open(scores, 'rb') as f:
    scores = pkl.load(f)  # Directly from multiobject.py

# Parameters
T0 = 11  # System memory
R = 5
eps = 1  # Noise variance
obj = 5
metric = 0  # if 0: JBLD, if 1: JKL
W = 3  # Smoothing window length
slow = False  # If true: Slow(but Precise), if false: Fast
norm = True  # If true: Norm, if false: MSE

metric_name = ['JBLD', 'JKL']
scores = scores[obj]
target_pos = target_pos_dict[obj]  # list of length 154. target_pod[f] is (2,)
target_sz = target_sz_dict[obj]


T = len(target_pos)
tin = 0
tfin = 60  #len(loc)

tracker = TrackerDyn_2(T0=T0, R=R,  W=W, noise=eps, metric=metric, slow=slow, norm=norm)
for f in range(tin, tfin):  # T
    print('-----Frame:', f, '-----')
    tracker.update(target_pos[f], target_sz[f], scores[f])


print(tracker.buffer_pos)
# plot_jbld_eta_score(tracker_pred, tracker_gt, obj, norm, slow, tin, tfin)


