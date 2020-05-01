import numpy as np
import torch
import pickle as pkl
from TrackerDyn import TrackerDyn
import matplotlib.pyplot as plt


# Tracker data
pred = '/Users/marinaalonsopoal/Desktop/dict_pred.obj'
gt = '/Users/marinaalonsopoal/Desktop/centr_gt'
locs = '/Users/marinaalonsopoal/Desktop/locations_SMOT_acrobats.obj'

with open(pred, 'rb') as f:
    preds = pkl.load(f)  # Manipulated on conv_cand_to_centr.py
with open(gt, 'rb') as f:
    gts = pkl.load(f)  # Manipulated on COLAB
with open(locs, 'rb') as f:
    locs = pkl.load(f)  # Directly from multiobject.py


# Parameters
T0 = 4
eps = 0.0001
obj = 5

pred = preds[obj]  # np.ndarray (154, 2)
gt = gts[obj]  # np.ndarray (154, 2)
loc = locs[obj]  # list(154)
# locs_1[1] list(num_candidates for frame 1)
# locs_1[1][0][0]  np.ndarray (8,)

# NOTE: Al SiamMask anirà al revés: 1. Frame 2. Objecte, però en teoria no afecta

tracker = TrackerDyn(T0=T0, noise=eps)
# TODO: Pensar que fer amb la (re)inicialització

jblds = np.zeros(len(loc))
for f in range(len(loc)):
    print('-----Frame:', f, '-----')
    # De moment treballo només amb el candidat cand, si vull canviar, iterar
    cand = 0
    loca = loc[f][cand][0]
    tracker.update(loca)
    jblds[f] = tracker.jbld()

jblds = np.nan_to_num(jblds)
max_jbld = np.argmax(jblds)
centr_x = tracker.buffer_centr[:, 0]
centr_y = tracker.buffer_centr[:, 1]



# NOTE: Visualization
col = ['g', 'm', 'y', 'b', 'r']
s_gt = 50
s_pred = 25

frames = np.arange(len(loc))
fig, (ax1, ax2, ax3) = plt.subplots(3)
fig.suptitle('Object ' + str(obj))
ax1.set_ylabel('Coordinate x')
ax2.set_ylabel('Coordinate y')
ax3.set_ylabel('JKL Distance')
ax3.set_xlabel('frame')
ax3.set_title('Joint (x,y) distance with T0=' + str(T0))

ax1.axvline(x=max_jbld, linewidth=2, color='k', zorder=0)
ax2.axvline(x=max_jbld, linewidth=2, color='k', zorder=0)
ax3.axvline(x=max_jbld, linewidth=2, color='k', zorder=0)

ax1.scatter(frames, centr_x, c=col[obj - 1], s=s_pred, edgecolors='k', alpha=1 )
ax1.scatter(frames, gt[:, 0], c=col[obj - 1], s=s_gt, alpha=0.2)

ax2.scatter(frames, centr_y, c=col[obj - 1], s=s_pred, edgecolors='k', alpha=1)
ax2.scatter(frames, gt[:, 1], c=col[obj - 1], s=s_gt, alpha=0.2)

ax3.plot(frames, jblds, c=col[obj - 1])

plt.show()