import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

frame_path = '/Users/marinaalonsopoal/Desktop/caminant/frame0.jpg'

im = np.array(Image.open(frame_path), dtype=np.uint8)


xleft, ytop, xright, ybottom = 435, 253, 483, 385

h = ybottom - ytop  # y axis is inverted
w = xright - xleft

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((xleft, ytop), w, h, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

# Scatter corners
ax.scatter(xleft, ybottom, s=50, c='yellow')
ax.scatter(xright, ybottom, s=50, c='green')
ax.scatter(xleft, ytop, s=50, c='orange')
ax.scatter(xright, ytop, s=50, c='blue')

plt.show()

print('(', xright, ',', ytop, ')')