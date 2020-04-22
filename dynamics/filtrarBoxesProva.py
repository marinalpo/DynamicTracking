import numpy as np
from shapely.geometry import asPolygon
import cv2
from utils_visualization import *


def compute_intersection_by_pairs(polygons, sorted_scores, intersection_th=0.5):
    "Computes "
    results = []
    for i in range(polygons.shape[0] // 2):
        pol_a = asPolygon(polygons[2*i, :, :])
        pol_b = asPolygon(polygons[2*i + 1, :, :])
        intersection_area = pol_a.intersection(pol_b)
        iou = intersection_area.area / (pol_a.area + pol_b.area)
        if iou >= intersection_th:
            if (sorted_scores[2*i] >= sorted_scores[2*i + 1]):
                results.append([polygons[2*i, :, :], sorted_scores[2*i]])
            else:
                results.append([polygons[2*i + 1, :, :], sorted_scores[2*i+1]])
        else:
            results.append([polygons[2*i, :, :], sorted_scores[2*i]])
            results.append([polygons[2*i + 1, :, :], sorted_scores[2*i+1]])
    return results



def filter_bboxes(rboxes, k):
    """
    rboxes: list, contains: [(np.array(polygon), score),(), ... , ()]
    """

    num_bxs = len(rboxes)

    c = 10 * len(rboxes)
    if num_bxs <= 1:
        return rboxes
    if num_bxs == k:
        return rboxes

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
    intersection_th = 0.25  # If 2 boxes overlap more than 0.4 the one with max score will remain

    if ((xmax - xmin) >= (ymax - ymin)):
        # Make pairs according to x axis (indexes_x)
        polygons = polygons[indexes_x]
        sorted_scores = scores[indexes_x]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores, intersection_th)
    else:
        # Make pairs according to y axis (indexes_y)
        polygons = polygons[indexes_y]
        sorted_scores = scores[indexes_y]
        rboxes = compute_intersection_by_pairs(polygons, sorted_scores, intersection_th)
    if (len(rboxes) != 1 and len(rboxes) > k):
        c -= 1
        rboxes = filter_bboxes(rboxes, k)

    return rboxes

def filter_k_boxes(rboxes, k):
    winner = rboxes[0]
    rboxes_filt = filter_bboxes(rboxes, k)
    for i, (box, score) in enumerate(rboxes_filt):
        if (box == winner[0]).all():
            print('delated duplicated winner')
            del rboxes_filt[i]
    rboxes_filt.insert(0, list(winner))
    return rboxes_filt

rboxes = np.load('/Users/marinaalonsopoal/Desktop/rboxes.npy')


# Parameters
k = 4
height = 1000
width = 2*height
colors = [(0, 0, 0), (0, 0, 255), (0, 213, 255), (0, 255, 76), (255, 255, 0), (255, 153, 0), (255, 0, 68), (221, 0, 255)]
colors = create_colormap_hsv(len(rboxes))
colors [5] = (255, 255, 0)
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
    cv2.polylines(canvas, [location], True, colors[box], 2)

crop = canvas[min(ymin)-margin:max(ymax)+margin, min(xmin)-margin:max(xmax)+margin]
for box in range(len(rboxes)):
    cv2.putText(canvas,str(box), (int(min(xmin) - margin/2 + 6*box), min(ymin)), font, font_size, colors[box], 1, cv2.LINE_AA)
cv2.imshow('Original rboxes', crop)



canvas = 255*np.ones((height, width, 3), np.uint8)
rboxes_filt = filter_k_boxes(rboxes, k)




for box_filt in range(len(rboxes_filt)):
    location1 = rboxes_filt[box_filt][0].flatten()
    location = np.int0(location1).reshape((-1, 1, 2))
    cv2.polylines(canvas, [location], True, colors[box_filt], 2)

crop = canvas[min(ymin)-margin:max(ymax)+margin, min(xmin)-margin:max(xmax)+margin]
for box_filt in range(len(rboxes_filt)):
    cv2.putText(canvas, str(box_filt), (int(min(xmin) - margin/2 + 6*box_filt), min(ymin)), font, font_size, colors[box_filt], 1, cv2.LINE_AA)

cv2.imshow('Filtered rboxes', crop)
cv2.waitKey()


print('Len original rboxes:', len(rboxes))
print('Primera bbox bona:', rboxes[0][0].flatten())
print('Number of required candidates:', k)
print('Len filt rboxes:', len(rboxes_filt))
print('Primera bbox bona:', rboxes_filt[0][0].flatten())