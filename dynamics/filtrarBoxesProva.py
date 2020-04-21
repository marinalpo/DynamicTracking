import numpy as np
from shapely.geometry import asPolygon
import cv2
from utils_visualization import *


def compute_intersection_by_pairs(polygons, sorted_scores, intersection_th=0.5):
    "Computes "
    results = []
    for i in range(polygons.shape[0] // 2):
        pol_a = asPolygon(polygons[i, :, :])
        pol_b = asPolygon(polygons[i + 1, :, :])
        intersection_area = pol_a.intersection(pol_b)
        iou = intersection_area.area / (pol_a.area + pol_b.area)
        if iou >= intersection_th:
            if (sorted_scores[i] >= sorted_scores[i + 1]):
                results.append([polygons[i, :, :], sorted_scores[i]])
            else:
                results.append([polygons[i + 1, :, :], sorted_scores[i+1]])
        else:
            if (sorted_scores[i] >= sorted_scores[i + 1]):
                results.append([polygons[i, :, :], sorted_scores[i]])
            else:
                results.append([polygons[i + 1, :, :], sorted_scores[i+1]])
    return results


def filter_bboxes(rboxes, k, c=10):
    """
    rboxes: list, contains: [(np.array(polygon), score),(), ... , ()]
    """
    num_bxs = len(rboxes)
    if c == 0:  # Evita bucle infinit
        return rboxes
    if num_bxs <= 1:
        return rboxes
    if num_bxs % 2 != 0:
        # We compare by pairs so we need an odd number of boxes
        del rboxes[-1]
        num_bxs = len(rboxes)

    polygons = np.zeros((num_bxs, 4, 2))  # (num_bxs, 8 (points of a polygon, 2 (coordinates of each point)))
    scores = np.zeros(num_bxs)
    for i, (box, score) in enumerate(rboxes):
        box_mat = box.reshape(4, 2)
        polygons[i, :, :] = box_mat
        scores[i] = score
    # Compute centroids
    centroids = np.mean(polygons, axis=1)
    indexes_x = np.argsort(centroids[:, 0])  # Ascending order
    indexes_y = np.argsort(centroids[:, 1])
    centroids_x = centroids[indexes_x, 0]
    centroids_y = centroids[indexes_y, 1]

    xmax, xmin = centroids_x[-1], centroids_x[0]
    ymax, ymin = centroids_y[-1], centroids_y[0]
    intersection_th = 0.7  # If 2 boxes overlap more than 0.4 the one with max score will remain

    if ((xmax - xmin) >= (ymax - ymin)):
        # Make pairs according to x axis (indexes_x)
        polygons = polygons[indexes_x]
        sorted_scores = scores[indexes_x]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores)
    else:
        # Make pairs according to y axis (indexes_y)
        polygons = polygons[indexes_y]
        sorted_scores = scores[indexes_y]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores)

    if (len(rboxes) != 1 and len(rboxes) > k):
        c -= 1
        rboxes = filter_bboxes(rboxes, k, c)

    return rboxes


rboxes = np.load('/Users/marinaalonsopoal/Desktop/rboxes.npy')


# Parameters
cand = 3
height = 1000
width = 2*height
colors = create_colormap_hsv(len(rboxes))
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.3
margin = 20

canvas = 255*np.ones((height, width, 3), np.uint8)
xmax = []
xmin = []
ymax = []
ymin = []
for box in range(len(rboxes)):
    location1 = rboxes[box][0].flatten()
    location = np.int0(location1).reshape((-1, 1, 2))
    xmax.append(location[:, :, 0].max())
    xmin.append(location[:, :, 0].min())
    ymax.append(location[:, :, 1].max())
    ymin.append(location[:, :, 1].min())
    cv2.polylines(canvas, [location], True, colors[box], 1)

crop = canvas[min(ymin)-margin:max(ymax)+margin, min(xmin)-margin:max(xmax)+margin]
for box in range(len(rboxes)):
    cv2.putText(canvas,str(box), (int(min(xmin) - margin/2 + 6*box), min(ymin)), font, font_size,colors[box], 1, cv2.LINE_AA)
cv2.imshow('Original rboxes', crop)

print('Len original rboxes:', len(rboxes))
print('Primera bbox bona:', rboxes[0][0].flatten())

canvas = 255*np.ones((height, width, 3), np.uint8)
rboxes_filt = filter_bboxes(rboxes, cand, c=10 * len(rboxes))

print('Number of required candidates:', cand)
print('Len filt rboxes:', len(rboxes_filt))
print('Primera bbox bona:', rboxes[0][0].flatten())

for box in range(len(rboxes_filt)):
    location1 = rboxes_filt[box][0].flatten()
    location = np.int0(location1).reshape((-1, 1, 2))
    xmax.append(location[:, :, 0].max())
    xmin.append(location[:, :, 0].min())
    ymax.append(location[:, :, 1].max())
    ymin.append(location[:, :, 1].min())
    cv2.polylines(canvas, [location], True, colors[box], 1)

crop = canvas[min(ymin)-margin:max(ymax)+margin, min(xmin)-margin:max(xmax)+margin]
for box in range(len(rboxes_filt)):
    cv2.putText(canvas, str(box), (int(min(xmin) - margin/2 + 6*box), min(ymin)), font, font_size,colors[box], 1, cv2.LINE_AA)

cv2.imshow('Filtered rboxes', crop)
cv2.waitKey()
