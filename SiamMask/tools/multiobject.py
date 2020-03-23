import glob
import numpy as np
import cv2
import pickle
import argparse
from tools.test_2_nostre import *
from utils.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
from shapely.geometry import asPolygon
from custom import Custom

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/balls', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

# PARAMETERS
colors = ['red', 'yellow', 'green', 'blue', 'orange', 'purple']
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
drawMask = False

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
                results.append([polygons[i + 1, :, :], sorted_scores[i]])
        else:
            if (sorted_scores[i] >= sorted_scores[i + 1]):
                results.append([polygons[i, :, :], sorted_scores[i]])
            else:
                results.append([polygons[i + 1, :, :], sorted_scores[i]])
    return results


def filter_bboxes(rboxes, k, c=10):
    """
    rboxes: list, contains: [(np.array(polygon), score),(), ... , ()]
    """
    num_bxs = len(rboxes)
    if (c == 0):  # Evita bucle infinit
        return rboxes
    if (num_bxs <= 1):
        return rboxes
    if (num_bxs % 2 != 0):
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
    intersection_th = 0.9  # If 2 boxes overlap more than 0.4 the one with max score will remain

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




if __name__ == '__main__':

    toc = 0  # Timer

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])

    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    # TODO: Change frames
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))[000:25]
    ims = [cv2.imread(imf) for imf in img_files]
    im_ori = ims[0]  # First frame

    # TODO: Write initial bboxes
    init_rects = [(107, 445, 62, 100), (236, 211, 104, 67), (187, 621, 88, 60) ]  # ants1 (ant1, ant2 and ant3)
    n_obj = len(init_rects)  # Number of objects being tracked
    target_pos = np.zeros((n_obj, 2))  # Targets centroids in the first frame
    target_sz = np.zeros((n_obj, 2))  # Targets height and width

    for i in range(n_obj):
        x, y, w, h = init_rects[i]
        target_pos[i, :] = np.array([x + w / 2, y + h / 2])
        target_sz[i, :] = np.array([w, h])

    all_centroids = {}

    for f, im in enumerate(ims):
        tic = cv2.getTickCount()

        if f == 0:  # Initialize tracker
            list_states = []
            for i in range(n_obj):
                dict_key = 'object' + str(i)  # object1, object2, ...
                all_centroids[dict_key] = []
                state = siamese_init(im_ori, target_pos[i, :], target_sz[i, :], siammask, cfg['hp'], device=device)  # init tracker
                list_states.append(state)

        elif f > 0:  # tracking
            for i in range(n_obj):
                dict_key = 'object' + str(i)  # object1, object2, ...
                frame_boxes = []
                state_obj = list_states[i]
                state_init = siamese_init(im_ori, target_pos[i, :], target_sz[i, :], siammask, cfg['hp'], device=device)
                state, bboxes, rboxes = siamese_track(state_obj, im, mask_enable=True, refine_enable=True,
                                                       device=device)  # track
                list_states[i] = state

                rboxes = filter_bboxes(rboxes, 1, c=10 * len(rboxes))  # Return c bounding boxes
                # dibuixar nomes la bbox filtrada si esta fora de la bona (state)
                locaux_bona = np.int0(state['ploygon'].flatten()).reshape((-1, 2))
                locaux_filt = rboxes[0][0]
                pol_a = asPolygon(locaux_bona)
                pol_b = asPolygon(locaux_filt)
                intersection_area = pol_a.intersection(pol_b)
                if ((intersection_area.area / (pol_a.area + pol_b.area)) <= 0.2):
                    for i in range(len(rboxes)):
                        location = rboxes[i][0].flatten()
                        location = np.int0(location).reshape((-1, 1, 2))
                        cv2.polylines(im, [location], True, (255, 255, 0), 1)
                        traj = np.average(location, axis=0)[0]
                        frame_boxes.append([traj])
                        cv2.circle(im, (int(traj[0]), int(traj[1])), 3, (255, 255, 0), 2)

                location = state['ploygon'].flatten()
                if drawMask:
                    mask = state['mask'] > state['p'].seg_thr
                    im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]

                laloc = np.int0(location).reshape((-1, 1, 2))
                traj = np.int0(np.average(laloc, axis=0)[0])
                frame_boxes.append([traj])
                all_centroids[dict_key].append(frame_boxes)

                cv2.circle(im, (traj[0], traj[1]), 3, colors[i], 2)
                cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, colors[i], 3)
                # cv2.putText(im, str(state['score']), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))

            # TODO: Omplir amb el video que s'esta tractant
            # Save results
            cv2.imwrite('/data/Marina/ants1/results/' + str(f) + '.jpeg', im)

            # TODO: tracking
            # (x,y) = tracker.decide(points) on points es una llista de candidats
        toc += cv2.getTickCount() - tic

    # TODO: canviar nom
    with open('/data/Marina/ants1/points/centroids_ant1.obj', 'wb') as fil:
        pickle.dump(all_centroids, fil)


    toc /= cv2.getTickFrequency()
    fps = f / toc

    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
