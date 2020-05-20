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


def plot_centr_pos_and_vel(tracker_pred, tracker_gt, obj, name, T0, eps):

    # Parameters
    col = ['g', 'm', 'y', 'b', 'r']
    s_gt = 50
    s_pred = 25

    # Extract centroids data from GT and Predictions
    centr_x = tracker_pred.buffer_centr_smo[:, 0]
    centr_y = tracker_pred.buffer_centr_smo[:, 1]
    centr_x_vel = tracker_pred.buffer_centr_vel_smo[:, 0]
    centr_y_vel = tracker_pred.buffer_centr_vel_smo[:, 1]

    centr_x_gt = tracker_gt.buffer_centr[:, 0]
    centr_y_gt = tracker_gt.buffer_centr[:, 1]

    dist_centr_x = tracker_pred.dist_centr_smo[:, 0]
    dist_centr_y = tracker_pred.dist_centr_smo[:, 1]
    dist_centr_x_vel = tracker_pred.dist_centr_vel_smo[:, 0]
    dist_centr_y_vel = tracker_pred.dist_centr_vel_smo[:, 1]

    centr_x = tracker_pred.buffer_centr[:, 0]
    centr_y = tracker_pred.buffer_centr[:, 1]
    centr_x_vel = tracker_pred.buffer_centr_vel[:, 0]
    centr_y_vel = tracker_pred.buffer_centr_vel[:, 1]

    centr_x_gt = tracker_gt.buffer_centr[:, 0]
    centr_y_gt = tracker_gt.buffer_centr[:, 1]

    dist_centr_x = tracker_pred.dist_centr[:, 0]
    dist_centr_y = tracker_pred.dist_centr[:, 1]
    dist_centr_x_vel = tracker_pred.dist_centr_vel[:, 0]
    dist_centr_y_vel = tracker_pred.dist_centr_vel[:, 1]

    xmax = max(dist_centr_x)
    xmax_vel = max(dist_centr_x_vel)
    ymax = max(dist_centr_y)
    ymax_vel = max(dist_centr_y_vel)
    tot_max = max(xmax, ymax, xmax_vel, ymax_vel)

    frames = np.arange(len(tracker_pred.dist_centr_joint))

    fig, ax = plt.subplots(4, 2)
    fig.tight_layout()

    fig.suptitle('Object ' + str(obj))

    ax[0, 0].set_title('POSITION Centroid x')
    ax[0, 0].scatter(frames, centr_x, c='r', s=s_pred, edgecolors='k', alpha=1, label='Predictions')
    ax[0, 0].scatter(frames, centr_x_gt, c='r', s=s_gt, alpha=0.2, label='GT')
    ax[0, 0].legend()
    ax[0, 0].set_ylabel('pixel')

    ax[0, 1].set_title('POSITION Centroid y')
    ax[0, 1].scatter(frames, centr_y, c='g', s=s_pred, edgecolors='k', alpha=1, label='Predictions')
    ax[0, 1].scatter(frames, centr_y_gt, c='g', s=s_gt, alpha=0.2, label='GT')
    ax[0, 1].legend()
    ax[0, 1].set_ylabel('pixel')

    ax[1, 0].set_title(name + ' distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 1].set_title(name + ' distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 0].plot(frames, dist_centr_x, c='r', alpha=1)
    ax[1, 1].plot(frames, dist_centr_y, c='g', alpha=1)
    ax[1, 0].set_ylabel('distance')
    ax[1, 1].set_ylabel('distance')
    ax[1, 1].set_ylim([0, tot_max + tot_max * 0.1])
    ax[1, 0].set_ylim([0, tot_max + tot_max * 0.1])

    ax[2, 0].set_title('VELOCITY Centroid x')
    ax[2, 0].scatter(frames, centr_x_vel, c='r', s=s_pred, edgecolors='k', alpha=1)
    ax[2, 0].plot(frames, centr_x_vel, c='r', alpha=0.2)
    ax[2, 0].set_ylabel('pixel')

    ax[2, 1].set_title('VELOCITY Centroid y')
    ax[2, 1].scatter(frames, centr_y_vel, c='g', s=s_pred, edgecolors='k', alpha=1)
    ax[2, 1].plot(frames, centr_y_vel, c='g', alpha=0.2)
    ax[2, 1].set_ylabel('pixel')

    ax[3, 0].set_title(name + ' distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[3, 1].set_title(name + ' distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[3, 0].plot(frames, dist_centr_x_vel, c='r', alpha=1)
    ax[3, 1].plot(frames, dist_centr_y_vel, c='g', alpha=1)
    ax[3, 0].set_xlabel('frame')
    ax[3, 1].set_xlabel('frame')
    ax[3, 0].set_ylabel('distance')
    ax[3, 1].set_ylabel('distance')
    ax[3, 1].set_ylim([0, tot_max + tot_max * 0.1])
    ax[3, 0].set_ylim([0, tot_max + tot_max * 0.1])

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


def plot_centr_and_jbld(tracker_pred, tracker_gt, scores):
    c = tracker_pred.buffer_centr
    c_gt = tracker_gt.buffer_centr
    jblds = tracker_pred.dist_centr_smo
    jblds_vel = tracker_pred.dist_centr_vel_smo
    jblds_2 = tracker_pred.dist_centr_smo_2
    W = tracker_pred.W
    eps = tracker_pred.noise
    T0 = tracker_pred.T0

    s_gt = 50
    s_pred = 25

    # Extract centroids data from GT and Predictions
    centr_x = c[:, 0]
    centr_y = c[:, 1]

    centr_x_gt = c_gt[:, 0]
    centr_y_gt = c_gt[:, 1]

    dist_centr_x = jblds[:, 0]
    dist_centr_x_vel_smo = jblds_vel[:, 0]
    dist_centr_x = 3*np.power(dist_centr_x, 2) + np.power(dist_centr_x_vel_smo, 2) -1*np.power(scores, 1) - 1 + jblds_2[:, 0]
    super_threshold_indices = dist_centr_x < 0
    dist_centr_x[super_threshold_indices] = 0
    dist_centr_y = jblds[:, 1]
    xmax = max(dist_centr_x)
    xmax_pos = dist_centr_x.argmax()
    ymax = max(dist_centr_y)
    ymax_pos = dist_centr_y.argmax()
    tot_max = max(xmax, ymax)

    frames = np.arange(len(centr_x))

    fig, ax = plt.subplots(2, 2)
    fig.tight_layout()
    ax[0, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[1, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[0, 0].grid(axis='x', zorder=1, alpha=0.4)
    ax[1, 0].grid(axis='x', zorder=1, alpha=0.4)

    ax[0, 0].set_title('Position Centroid x with smooth W=' + str(W))
    ax[0, 0].scatter(frames, centr_x, c='r', s=s_pred, edgecolors='k', alpha=1, label='Predictions', zorder=3)
    ax[0, 0].scatter(frames, centr_x_gt, c='r', s=s_gt, alpha=0.2, label='GT', zorder=3)
    ax[0, 0].legend()
    ax[0, 0].set_ylabel('pixel')
    ax[0, 0].axvline(xmax_pos, c='k', linestyle=':', zorder=2)

    ax[0, 1].set_title('Position Centroid y with smooth W=' + str(W))
    ax[0, 1].scatter(frames, centr_y, c='g', s=s_pred, edgecolors='k', alpha=1, label='Predictions', zorder=3)
    ax[0, 1].scatter(frames, centr_y_gt, c='g', s=s_gt, alpha=0.2, label='GT', zorder=3)
    ax[0, 1].legend()
    ax[0, 1].axvline(ymax_pos, c='k', linestyle=':', zorder=2)
    ax[0, 1].set_ylabel('pixel')

    ax[1, 0].set_title('JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 1].set_title('JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 0].plot(frames, dist_centr_x, c='r', alpha=1)
    ax[1, 1].plot(frames, dist_centr_y, c='g', alpha=1)
    ax[1, 0].set_ylabel('distance')
    ax[1, 1].set_ylabel('distance')
    ax[1, 1].set_ylim([0, tot_max + tot_max * 0.1])
    # ax[1, 0].set_ylim([0, tot_max + tot_max * 0.1])
    ax[1, 0].axvline(xmax_pos, c='k', linestyle=':', zorder=2)
    ax[1, 1].axvline(ymax_pos, c='k', linestyle=':', zorder=2)

    plt.show()


def plot_centr_and_jbld_2(tracker_pred, tracker_gt):
    W = tracker_pred.W
    eps = tracker_pred.noise
    T0 = tracker_pred.T0

    s_gt = 25
    s_pred = 25

    # Extract centroids data from GT and Predictions
    centr_x = tracker_pred.buffer_centr[:, 0]
    centr_y = tracker_pred.buffer_centr[:, 1]

    centr_x_smo = tracker_pred.buffer_centr_smo[:, 0]
    centr_y_smo = tracker_pred.buffer_centr_smo[:, 1]

    centr_x_gt = tracker_gt.buffer_centr[:, 0]
    centr_y_gt = tracker_gt.buffer_centr[:, 1]

    dist_centr_x = tracker_pred.dist_centr[:, 0]
    dist_centr_y = tracker_pred.dist_centr[:, 1]

    dist_centr_x_smo = tracker_pred.dist_centr_smo[:, 0]
    dist_centr_y_smo = tracker_pred.dist_centr_smo[:, 1]


    xmax = max(dist_centr_x)
    xmax_smo = max(dist_centr_x_smo)
    ymax = max(dist_centr_y)
    ymax_smo = max(dist_centr_y_smo)
    tot_max = max(xmax, ymax, xmax_smo, ymax_smo)

    frames = np.arange(len(centr_x))

    fig, ax = plt.subplots(3, 2)
    fig.tight_layout()
    ax[0, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[1, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[0, 0].grid(axis='x', zorder=1, alpha=0.4)
    ax[1, 0].grid(axis='x', zorder=1, alpha=0.4)
    ax[2, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[2, 0].grid(axis='x', zorder=1, alpha=0.4)

    ax[0, 0].set_title('Position Centroid x with smooth W=' + str(W))
    ax[0, 0].scatter(frames, centr_x_smo, c='r', s=s_pred, edgecolors='k', alpha=1, label='Smoothed', zorder=3)
    ax[0, 0].scatter(frames, centr_x, c='r', s=s_pred, edgecolors='k', alpha=0.4, label='Predictions', zorder=3)
    ax[0, 0].scatter(frames, centr_x_gt, c='r', s=s_gt, alpha=0.05, label='GT', zorder=3)
    ax[0, 0].legend()
    ax[0, 0].set_ylabel('pixel')

    ax[0, 1].set_title('Position Centroid y with smooth W=' + str(W))
    ax[0, 1].scatter(frames, centr_y_smo, c='g', s=s_pred, edgecolors='k', alpha=1, label='Smoothed', zorder=3)
    ax[0, 1].scatter(frames, centr_y, c='g', s=s_pred, edgecolors='k', alpha=0.4, label='Predictions', zorder=3)
    ax[0, 1].scatter(frames, centr_y_gt, c='g', s=s_gt, alpha=0.05, label='GT', zorder=3)
    ax[0, 1].legend()
    ax[0, 1].set_ylabel('pixel')

    ax[1, 0].set_title('JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 1].set_title('JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 0].plot(frames, dist_centr_x, c='r', alpha=1)
    ax[1, 1].plot(frames, dist_centr_y, c='g', alpha=1)
    ax[1, 0].set_ylabel('distance')
    ax[1, 1].set_ylabel('distance')
    ax[1, 1].set_ylim([0, tot_max + tot_max * 0.1])
    ax[1, 0].set_ylim([0, tot_max + tot_max * 0.1])


    ax[2, 0].set_title('JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps) + ' on SMOOTHED data')
    ax[2, 1].set_title('SMOOTHED JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps) + ' on SMOOTHED data')
    ax[2, 0].plot(frames, dist_centr_x_smo, c='r', alpha=1)
    ax[2, 1].plot(frames, dist_centr_y_smo, c='g', alpha=1)
    ax[2, 0].set_ylabel('distance')
    ax[2, 1].set_ylabel('distance')
    ax[2, 1].set_ylim([0, tot_max + tot_max * 0.1])
    ax[2, 0].set_ylim([0, tot_max + tot_max * 0.1])
    plt.show()


def plot_centr_and_jbld_3(tracker_pred, tracker_gt):
    W = tracker_pred.W
    eps = tracker_pred.noise
    T0 = tracker_pred.T0

    s_gt = 25
    s_pred = 25

    # Extract centroids data from GT and Predictions
    centr_x = tracker_pred.buffer_centr[:, 0]
    centr_y = tracker_pred.buffer_centr[:, 1]

    centr_x_gt = tracker_gt.buffer_centr[:, 0]
    centr_y_gt = tracker_gt.buffer_centr[:, 1]

    dist_centr_x = tracker_pred.dist_centr[:, 0]
    dist_centr_y = tracker_pred.dist_centr[:, 1]

    dist_centr_x_2 = tracker_pred.dist_centr_2[:, 0]
    dist_centr_y_2 = tracker_pred.dist_centr_2[:, 1]


    xmax = max(dist_centr_x)
    xmax_2 = max(dist_centr_x_2)
    ymax = max(dist_centr_y)
    ymax_2 = max(dist_centr_y_2)
    tot_max = max(xmax, ymax, xmax_2, ymax_2)

    frames = np.arange(len(centr_x))

    fig, ax = plt.subplots(3, 2)
    fig.tight_layout()
    ax[0, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[1, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[0, 0].grid(axis='x', zorder=1, alpha=0.4)
    ax[1, 0].grid(axis='x', zorder=1, alpha=0.4)
    ax[2, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[2, 0].grid(axis='x', zorder=1, alpha=0.4)

    ax[0, 0].set_title('Position Centroid x')
    ax[0, 0].scatter(frames, centr_x, c='r', s=s_pred, edgecolors='k', alpha=1, label='Predictions', zorder=3)
    ax[0, 0].scatter(frames, centr_x_gt, c='r', s=s_gt, alpha=0.2, label='GT', zorder=3)
    ax[0, 0].legend()
    ax[0, 0].set_ylabel('pixel')

    ax[0, 1].set_title('Position Centroid y')
    ax[0, 1].scatter(frames, centr_y, c='g', s=s_pred, edgecolors='k', alpha=1, label='Predictions', zorder=3)
    ax[0, 1].scatter(frames, centr_y_gt, c='g', s=s_gt, alpha=0.2, label='GT', zorder=3)
    ax[0, 1].legend()
    ax[0, 1].set_ylabel('pixel')

    ax[1, 0].set_title('Sliding Window JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 1].set_title('Sliding Window JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 0].plot(frames, dist_centr_x, c='r', alpha=1)
    ax[1, 1].plot(frames, dist_centr_y, c='g', alpha=1)
    ax[1, 0].set_ylabel('distance')
    ax[1, 1].set_ylabel('distance')
    ax[1, 1].set_ylim([0, tot_max + tot_max * 0.1])
    ax[1, 0].set_ylim([0, tot_max + tot_max * 0.1])


    ax[2, 0].set_title('Increasing Window JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[2, 1].set_title('Increasing Window JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[2, 0].plot(frames, dist_centr_x_2, c='r', alpha=1)
    ax[2, 1].plot(frames, dist_centr_y_2, c='g', alpha=1)
    ax[2, 0].set_ylabel('distance')
    ax[2, 1].set_ylabel('distance')
    ax[2, 1].set_ylim([0, tot_max + tot_max * 0.1])
    ax[2, 0].set_ylim([0, tot_max + tot_max * 0.1])
    plt.show()


def plot_centr_and_jbld_4(tracker_pred, tracker_gt, scores):
    W = tracker_pred.W
    eps = tracker_pred.noise
    T0 = tracker_pred.T0

    s_gt = 25
    s_pred = 25

    # Extract centroids data from GT and Predictions
    centr_x = tracker_pred.buffer_centr[:, 0]
    centr_y = tracker_pred.buffer_centr[:, 1]

    centr_x_gt = tracker_gt.buffer_centr[:, 0]
    centr_y_gt = tracker_gt.buffer_centr[:, 1]

    dist_centr_x = tracker_pred.dist_centr[:, 0]
    dist_centr_y = tracker_pred.dist_centr[:, 1]

    dist_centr_x_2 = tracker_pred.dist_centr_2[:, 0]
    dist_centr_y_2 = tracker_pred.dist_centr_2[:, 1]


    xmax = max(dist_centr_x)
    xmax_2 = max(dist_centr_x_2)
    ymax = max(dist_centr_y)
    ymax_2 = max(dist_centr_y_2)
    tot_max = max(xmax, ymax, xmax_2, ymax_2)

    frames = np.arange(len(centr_x))

    fig, ax = plt.subplots(3, 2)
    fig.tight_layout()
    ax[0, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[1, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[0, 0].grid(axis='x', zorder=1, alpha=0.4)
    ax[1, 0].grid(axis='x', zorder=1, alpha=0.4)
    ax[2, 1].grid(axis='x', zorder=1, alpha=0.4)
    ax[2, 0].grid(axis='x', zorder=1, alpha=0.4)

    ax[0, 0].set_title('Position Centroid x')
    ax[0, 0].scatter(frames, centr_x, c='r', s=s_pred, edgecolors='k', alpha=1, label='Predictions', zorder=3)
    ax[0, 0].scatter(frames, centr_x_gt, c='r', s=s_gt, alpha=0.2, label='GT', zorder=3)
    ax[0, 0].legend()
    ax[0, 0].set_ylabel('pixel')

    ax[0, 1].set_title('Position Centroid y')
    ax[0, 1].scatter(frames, centr_y, c='g', s=s_pred, edgecolors='k', alpha=1, label='Predictions', zorder=3)
    ax[0, 1].scatter(frames, centr_y_gt, c='g', s=s_gt, alpha=0.2, label='GT', zorder=3)
    ax[0, 1].legend()
    ax[0, 1].set_ylabel('pixel')

    ax[1, 0].set_title('JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 1].set_title('JBLD distance with T0=' + str(T0) + ' and noise=' + str(eps))
    ax[1, 0].plot(frames, dist_centr_x, c='r', alpha=1)
    ax[1, 1].plot(frames, dist_centr_y, c='g', alpha=1)
    ax[1, 0].set_ylabel('distance')
    ax[1, 1].set_ylabel('distance')
    ax[1, 1].set_ylim([0, tot_max + tot_max * 0.1])
    ax[1, 0].set_ylim([0, tot_max + tot_max * 0.1])


    ax[2, 0].set_title('Negative Tracker Score')
    ax[2, 1].set_title('Negative Tracker Score')

    ax[2, 0].plot(frames, scores, c='k', alpha=1)
    ax[2, 1].plot(frames, scores, c='k', alpha=1)
    ax[2, 0].set_ylabel('distance')
    plt.show()