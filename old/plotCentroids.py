import torch
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
# device = torch.device('cuda:0')

centroids = '/data/Marina/ants1/points/centroids_ants1.obj'
groundtruth = '/data/Marina/ants1/groundtruth/groundtruth.txt'

with open(centroids, 'rb') as f:
    data = pkl.load(f)

gt = np.loadtxt(groundtruth, delimiter=',')  # shape: (n_frames, 8)

size = 40
a = 1
count = 0
tini = 110
tfin = 150

fig, ax = plt.subplots(2, 1, sharey=True)

for t, points in enumerate(data):

    if tfin >= t >= tini:
        ax[0].axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)
        ax[1].axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)

        cx = np.mean([gt[t, 0], gt[t, 2], gt[t, 4], gt[t, 6]])
        cy = np.mean([gt[t, 1], gt[t, 3], gt[t, 5], gt[t, 7]])

        if t == tini:
            ax[0].scatter(t, points[0][0][0], s=size, c='k', zorder=1, alpha=a, label='Chosen Centroid')
            ax[1].scatter(t, points[0][0][1], s=size, c='k', zorder=1, alpha=a)
            ax[0].scatter(t, cx, s=size, c='g', alpha=a, label='Ground Truth')
            ax[1].scatter(t, cy, s=size, c='g', alpha=a)
        else:
            ax[0].scatter(t, cx, s=size, c='g', alpha=a)
            ax[1].scatter(t, cy, s=size, c='g', alpha=a)
            if len(points) == 1:
                ax[0].scatter(t, points[0][0][0], s=size, c='k', zorder=1, alpha=a)
                ax[1].scatter(t, points[0][0][1], s=size, c='k', zorder=1, alpha=a)
            else:
                for c in range(len(points)):
                    if c == 0:
                        ax[0].scatter(t, points[c][0][0], s=size, c='k', zorder=1, alpha=a)
                        ax[1].scatter(t, points[c][0][1], s=size, c='k', zorder=1, alpha=a)
                    else:
                        if count == 0:
                            ax[0].scatter(t, points[c][0][0], s=size, c='r', zorder=1, alpha=a, label='Discarded candidate')
                            ax[1].scatter(t, points[c][0][1], s=size, c='r', zorder=1, alpha=a)
                            count = count+1
                        else:
                            ax[0].scatter(t, points[c][0][0], s=size, c='r', zorder=1, alpha=a)
                            ax[1].scatter(t, points[c][0][1], s=size, c='r', zorder=1, alpha=a)



ax[0].legend()
ax[0].set_title('X coordinate')
ax[1].set_title('Y coordinate')
fig.suptitle('Candidate Centroids and Ground Truth\n VOT18 ants1 Sequence Crop', fontsize=12, color='b')
plt.xlabel('Frames')
plt.show()