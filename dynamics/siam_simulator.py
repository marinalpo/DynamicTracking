import numpy as np
import torch
import pickle as pkl
from TrackerDyn import TrackerDyn
import matplotlib.pyplot as plt


# Tracker data
gt = '/Users/marinaalonsopoal/Desktop/centr_gt'
locs_gt = '/Users/marinaalonsopoal/Desktop/location_gt'
locs_pred = '/Users/marinaalonsopoal/Desktop/locations_SMOT_acrobats.obj'

with open(locs_gt, 'rb') as f:
    locs_gt = pkl.load(f)  # Manipulated on conv_cand_to_centr.py
with open(gt, 'rb') as f:
    gts = pkl.load(f)  # Manipulated on COLAB
with open(locs_pred, 'rb') as f:
    locs_pred = pkl.load(f)  # Directly from multiobject.py


# Parameters
T0 = 4
eps = 0.001
obj = 4
metric = 0  # if 0: JBLD, if 1: JKL
metric_name = ['JBLD', 'JKL']

gt = gts[obj]  # np.ndarray (154, 2)
loc_gt = locs_gt[obj]  # np.ndarray (154, 8)

loc = locs_pred[obj]  # list(154)
# locs_1[1] list(num_candidates for frame 1)
# locs_1[1][0][0]  np.ndarray (8,)

# NOTE: Al SiamMask anirà al revés: 1. Frame 2. Objecte, però en teoria no afecta

tracker_pred = TrackerDyn(T0=T0, noise=eps, metric=metric)

jblds_pred = np.zeros(len(loc))

for f in range(len(loc)):
    print('-----Frame:', f, '-----')

    # De moment treballo només amb el candidat cand, si vull canviar, iterar
    cand = 0
    loca = loc[f][cand][0]
    tracker_pred.update(loca)
    jblds_pred[f] = tracker_pred.jbld()


jblds_pred = np.nan_to_num(jblds_pred)
max_jbld = np.argmax(jblds_pred)

centr_x = tracker_pred.buffer_centr[:, 0]
centr_y = tracker_pred.buffer_centr[:, 1]


# NOTE: Visualization
col = ['g', 'm', 'y', 'b', 'r']
s_gt = 50
s_pred = 25

frames = np.arange(len(loc))
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Object ' + str(obj))
ax1.set_ylabel('Coordinate x')
ax2.set_ylabel('Coordinate y')
ax3.set_ylabel('Distance')
ax3.set_xlabel('frame')
ax3.set_title('Joint (x,y) '+metric_name[metric]+' distance with T0=' + str(T0) + ' and noise=' + str(eps))

ax1.axvline(x=max_jbld, linewidth=2, color='k', zorder=0)
ax2.axvline(x=max_jbld, linewidth=2, color='k', zorder=0)
ax3.axvline(x=max_jbld, linewidth=2, color='k', zorder=0)

ax1.scatter(frames, centr_x, c=col[obj - 1], s=s_pred, edgecolors='k', alpha=1 )
ax1.scatter(frames, gt[:,0], c=col[obj - 1], s=s_gt, alpha=0.2)

ax2.scatter(frames, centr_y, c=col[obj - 1], s=s_pred, edgecolors='k', alpha=1)
ax2.scatter(frames, gt[:,1], c=col[obj - 1], s=s_gt, alpha=0.2)

ax3.plot(frames, jblds_pred, c=col[obj - 1])

plt.show()
