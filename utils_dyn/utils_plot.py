import numpy as np
import cv2

def get_location(gt):
    x, y, w, h = gt
    p1 = [x, y]
    p2 = [x + w, y]
    p3 = [x + w, y + h]
    p4 = [x, y + h]
    location = np.concatenate([p1, p2, p3, p4])
    location = np.int0(location).reshape((-1, 1, 2))  # shape (4, 1, 2)
    return location


def create_colormap_hsv(num_col):
    colors = []
    addGrayScale = False
    rounds = 3
    np.random.seed(2)
    if num_col <= 8:
        rounds = 1
        div = num_col
    elif num_col <= 20:
        div = 7
    else:
        div = int(np.ceil(num_col / 3))
        addGrayScale = True
    if addGrayScale:
        colors.append((220, 220, 220))
        colors.append((20, 20, 20))
        colors.append((127, 127, 127))
    portion = 180/div
    ss = [255, 255, 100]
    vs = [255, 100, 255]
    for j in range(rounds):
        s = ss[j]
        v = vs[j]
        for i in range(div):
            h = int(portion*i)
            color_hsv = np.uint8([[[h, s, v]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
            color_bgr = color_bgr[0][0]
            color_bgr = [int(cha) for cha in color_bgr]
            colors.append(tuple(color_bgr))
    colors = colors[0:num_col]
    np.random.shuffle(colors)
    return colors