import numpy as np
import matplotlib.pyplot as plt

def plot_all_dist(tracker_pred, dist, obj, T0, noise):
    plt.title(dist + ' Distances for object ' + str(obj) + ' with T0=' + str(T0) + ' and noise=' + str(noise))
    frames = np.arange(len(tracker_pred.dist_centr_joint))
    plt.plot(frames, tracker_pred.dist_centr[:, 0], linestyle='dashed', label='centr_x')
    plt.plot(frames, tracker_pred.dist_centr[:, 1], linestyle='dashed', label='centr_y')
    plt.plot(frames, tracker_pred.dist_centr_joint, linestyle='dashed', label='centr_joint')
    plt.plot(frames, tracker_pred.dist_loc[:, 0], label='loc_x1')
    plt.plot(frames, tracker_pred.dist_loc[:, 1], label='loc_y1')
    plt.plot(frames, tracker_pred.dist_loc[:, 2], label='loc_x2')
    plt.plot(frames, tracker_pred.dist_loc[:, 3], label='loc_y2')
    plt.plot(frames, tracker_pred.dist_loc[:, 4], label='loc_x3')
    plt.plot(frames, tracker_pred.dist_loc[:, 5], label='loc_y3')
    plt.plot(frames, tracker_pred.dist_loc[:, 6], label='loc_x4')
    plt.plot(frames, tracker_pred.dist_loc[:, 7], label='loc_y4')
    plt.plot(frames, tracker_pred.dist_loc_joint, label='loc_joint')
    plt.legend()
    plt.show()


def plot_centr_and_dist(tracker_pred, tracker_gt, name, obj, T0, eps):

    # Parameters
    col = ['g', 'm', 'y', 'b', 'r']
    s_gt = 50
    s_pred = 25

    # Extract centroids data from GT and Predictions
    centr_x = tracker_pred.buffer_centr[:, 0]
    centr_y = tracker_pred.buffer_centr[:, 1]
    centr_x_gt = tracker_gt.buffer_centr[:, 0]
    centr_y_gt = tracker_gt.buffer_centr[:, 1]

    frames = np.arange(len(tracker_pred.dist_centr_joint))

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)

    fig.suptitle('Object ' + str(obj) + '\n' + name + ' distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax1.set_ylabel('Coordinate x')
    ax2.set_ylabel('Coordinate y')
    ax3.set_ylabel('Distance')
    ax4.set_ylabel('Distance')
    ax4.set_xlabel('frame')

    ax1.scatter(frames, centr_x, c='r', s=s_pred, edgecolors='k', alpha=1, label='Predictions')
    ax1.scatter(frames, centr_x_gt, c='r', s=s_gt, alpha=0.2, label='GT')
    ax1.legend()

    ax2.scatter(frames, centr_y, c='g', s=s_pred, edgecolors='k', alpha=1, label='Predictions')
    ax2.scatter(frames, centr_y_gt, c='g', s=s_gt, alpha=0.2, label='GT')
    ax2.legend()

    ax3.plot(frames, tracker_pred.dist_centr[:, 0], c='r', label='coord x')
    ax3.plot(frames, tracker_pred.dist_centr[:, 1], c='g', label='coord y')
    ax3.legend()

    ax4.plot(frames, tracker_pred.dist_centr_joint, c='k', linewidth=2, label='joint (x,y)')
    ax4.legend()
    plt.show()


def plot_all(tracker_pred, tracker_gt, name, obj, T0, eps):

    locs_gt = tracker_gt.buffer_loc
    locs_pred = tracker_pred.buffer_loc
    dist_pred = tracker_pred.dist_loc
    # Extract centroids data from GT and Predictions
    centr_x = tracker_pred.buffer_centr[:, 0]
    centr_y = tracker_pred.buffer_centr[:, 1]
    centr_x_gt = tracker_gt.buffer_centr[:, 0]
    centr_y_gt = tracker_gt.buffer_centr[:, 1]

    col = ['r', 'g']
    s_gt = 50
    s_pred = 25

    frames = np.arange(len(tracker_pred.dist_centr_joint))

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(8, 3)
    fig.suptitle('Object ' + str(obj) + '\n' + name + ' distance with T0=' + str(T0) + ' and noise=' + str(eps))


    cords = ['x', 'y']
    for c in range(2):
        for i in range(4):
            f = fig.add_subplot(gs[2 * i, c])
            f.scatter(frames, locs_pred[:, 2 * i+c], c=col[c], s=s_pred, edgecolors='k', alpha=1, label='Predictions')
            f.scatter(frames, locs_gt[:, 2 * i+c], c=col[c], s=s_gt, alpha=0.2, label='GT')
            g = fig.add_subplot(gs[2 * i + 1, c])
            g.plot(frames, dist_pred[:, 2 * i+c], c=col[c])
            if c == 0:
                f.set_ylabel('Corner ' + str(i+1))
            if i == 0:
                f.set_title('Coordinate ' + cords[c])
            elif i == 3:
                g.set_xlabel('frame')

    ax1 = fig.add_subplot(gs[0:2, 2])
    ax1.scatter(frames, centr_x, c='r', s=s_pred, edgecolors='k', alpha=1, label='Predictions')
    ax1.scatter(frames, centr_x_gt, c='r', s=s_gt, alpha=0.2, label='GT')
    ax1.set_title('Centroids')
    ax1.legend()

    ax2 = fig.add_subplot(gs[2:4, 2])
    ax2.scatter(frames, centr_y, c='g', s=s_pred, edgecolors='k', alpha=1, label='Predictions')
    ax2.scatter(frames, centr_y_gt, c='g', s=s_gt, alpha=0.2, label='GT')
    ax2.legend()

    ax3 = fig.add_subplot(gs[4:6, 2])
    ax3.plot(frames, tracker_pred.dist_centr[:, 0], c='r', label='coord x')
    ax3.plot(frames, tracker_pred.dist_centr[:, 1], c='g', label='coord y')
    ax3.plot(frames, tracker_pred.dist_centr_joint, c='k', linewidth=2, label='joint (x,y)')
    ax3.legend()

    ax4 = fig.add_subplot(gs[6:8, 2])
    ax4.plot(frames, tracker_pred.dist_centr_joint, c='k', label='joint (x,y)')
    ax4.plot(frames, tracker_pred.dist_loc_joint, c='b', label='joint corners')
    ax4.set_ylabel('frame')
    ax4.legend()



    # ax3.plot(frames, tracker_pred.dist_centr[:, 0], c='r', label='coord x')
    # ax3.plot(frames, tracker_pred.dist_centr[:, 1], c='g', label='coord y')
    # ax3.plot(frames, tracker_pred.dist_centr_joint, c='k', label='joint', linewidth=3)
    # ax3.legend()
    plt.show()


def plot_locs_pred(tracker_gt, tracker_pred):
    frames = np.arange(len(tracker_pred.dist_centr_joint))
    locs_gt = tracker_gt.buffer_loc
    locs_pred = tracker_pred.buffer_loc
    dist_pred = tracker_pred.dist_loc
    fig = plt.figure()
    gs = fig.add_gridspec(4, 2)
    col = ['r', 'g']
    s_gt = 50
    s_pred = 25
    cords = ['x', 'y']
    for c in range(2):
        for i in range(4):
            f = fig.add_subplot(gs[i, c])
            f.scatter(frames, locs_pred[:, 2 * i+c], c=col[c], s=s_pred, edgecolors='k', alpha=1, label='Predictions')
            f.scatter(frames, locs_gt[:, 2 * i+c], c=col[c], s=s_gt, alpha=0.2, label='GT')
    plt.show()
