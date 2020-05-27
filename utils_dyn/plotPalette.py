import numpy as np
import cv2
from utils_visualization import *


# Load data
im = cv2.imread('/Users/marinaalonsopoal/Desktop/white_canvas.png')

# Parameters
width = 1500
im1 = cv2.resize(im, (width, 1000))
im2 = cv2.resize(im, (width, 1000))
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 1
num_colors = 20

colors_rgb = create_colormap_rgb(num_colors)
colors_hsv = create_colormap_hsv(num_colors)

portion = width/num_colors
for i in range(num_colors):
    x = int(portion*i+portion/2)
    cv2.line(im1, (x, 0), (x, 1000), color=colors_rgb[i], thickness=int(portion), shift=0)
    cv2.line(im2, (x, 0), (x, 1000), color=colors_hsv[i], thickness=int(portion), shift=0)
    # cv2.putText(im, str(i), (int(x-portion/3), 30), font, font_size, (255, 255, 255), 4, cv2.LINE_AA)

cv2.imshow('Color Palette RGB', im1)
cv2.imshow('Color Palette HSV', im2)

print(len(colors_rgb))
print(len(colors_hsv))
# cv2.imwrite('/Users/marinaalonsopoal/Desktop/colors/RGBq2.jpg', im)
cv2.waitKey()