import pickle as pkl
import matplotlib.pyplot as plt


def plot_vertical_lines(length, ax):
    for t in range(length):
        ax[0].axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)
        ax[1].axvline(x=t, color='gray', linestyle=':', linewidth=1, zorder=1)


def plot_centroids_and_candidates(value, ax, col, size, alpha):
    for t, points in enumerate(value):
        for c in range(len(points)):  # More than one point
            if c == 0:  # Decided point
                ax[0].scatter(t, points[0][0][0], edgecolor='k', c=col, zorder=3, s=size)
                ax[1].scatter(t, points[0][0][1], edgecolor='k', c=col, zorder=3, s=size)
            else:  # Discarded candidates
                ax[0].scatter(t, points[c][0][0], c=col, alpha=alpha, zorder=2, s=size)
                ax[1].scatter(t, points[c][0][1], c=col, alpha=alpha, zorder=2, s=size)

# Load Data
centroids = '/Users/marinaalonsopoal/Desktop/centroids_ant1.obj'
with open(centroids, 'rb') as f:
    data = pkl.load(f)

# Parameters
colors_cv2 = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
colors_plt = [[[int(comp/255) for comp in item]] for item in colors_cv2]

alpha = 0.5
size = 40
title = 'Centroids and discarded candidates evolution \nDataset: VOT2018 Sequence: ants1'

# Create figure and axes
fig, ax = plt.subplots(2, 1)

# Initialize counter
o = 1  # Object number
for key, value in data.items():  # For each object being tracked
    plot_centroids_and_candidates(value, ax, colors_plt[o - 1], size, alpha)
    o += 1
num_frames = len(value)
plot_vertical_lines(num_frames, ax)

fig.suptitle(title, fontsize=12, color='b')
ax[0].set_ylabel('X coordinate')
ax[1].set_ylabel('Y coordinate')
plt.xlabel('Frames')
plt.show()







