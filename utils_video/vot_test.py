import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

groundtruth = '/Users/marinaalonsopoal/Desktop/ants1/groundtruth.txt'
image = '/Users/marinaalonsopoal/Desktop/ants1/images/00000001.jpg'

gt = np.loadtxt(groundtruth, delimiter=',')  # shape: (n_frames, 8)
im = np.array(Image.open(image), dtype=np.uint8)


fig, ax = plt.subplots(1)
colors = ['yellow', 'red', 'blue', 'green']
gt_frame = gt[0, :]
centroid_x = np.mean([gt_frame[0], gt_frame[2], gt_frame[4], gt_frame[6]])
centroid_y = np.mean([gt_frame[1], gt_frame[3], gt_frame[5], gt_frame[7]])

for i in range(4):
    ax.scatter(gt_frame[2*i], gt_frame[2*i+1], s=10, c=colors[i])
ax.scatter(centroid_x, centroid_y, s=10, c='white')

# Display the image
ax.imshow(im)

# Create a Rectangle patch
# rect = patches.Rectangle((xleft, ytop), w, h, linewidth=1, edgecolor='r', facecolor='none')
#
# # Add the patch to the Axes
# ax.add_patch(rect)


plt.show()