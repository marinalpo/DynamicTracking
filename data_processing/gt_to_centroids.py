import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pickle

groundtruth = '/Users/marinaalonsopoal/Desktop/ants1/groundtruth.txt'
gt = np.loadtxt(groundtruth, delimiter=',')  # shape: (n_frames, 8)


n_frames = gt.shape[0]
centroids = []

for i in range(n_frames):
    cx = np.mean([gt[i, 0], gt[i, 2], gt[i, 4], gt[i, 6]])
    cy = np.mean([gt[i, 1], gt[i, 3], gt[i, 5], gt[i, 7]])
    point = (cx, cy)
    centroids.append(point)


with open('/Users/marinaalonsopoal/Desktop/ants1/gt_ants1.obj', 'wb') as fil:
    pickle.dump(centroids, fil)