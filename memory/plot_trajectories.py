import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

# Parameters
obja = 1
objb = 4
coord = 0
T = 140

target_pos_dict = '/Users/marinaalonsopoal/Desktop/Objects/positions/target_pos_dict.obj'
gta = np.load('/Users/marinaalonsopoal/Desktop/Objects/centr_gt_' + str(obja) + '.npy')  # shape: (154, 2)
gtb = np.load('/Users/marinaalonsopoal/Desktop/Objects/centr_gt_' + str(objb) + '.npy')
with open(target_pos_dict, 'rb') as f:
    target_pos_dict = pkl.load(f)  # Directly from multi_pred.py
preda_list = target_pos_dict[obja]  # list of length 154. target_pos[f] is (2,)
predb_list = target_pos_dict[objb]  # list of length 154. target_pos[f] is (2,)

preda = np.zeros(T)
predb = np.zeros(T)
frames = np.arange(T)
for f in range(0, T):
    preda[f] = preda_list[f][coord]
    predb[f] = predb_list[f][coord]
gta = gta[:T, coord]
gtb = gtb[:T, coord]

l = 8
a = 0.5

# plt.scatter(frames, gta, s=s, color=(1, 1, 1), edgecolors=(255/255, 181/255, 112/255), label='Object A GT')
# plt.scatter(frames, gtb, s=s, color=(1, 1, 1), edgecolors=(126/255, 166/255, 224/255), label='Object B GT')
plt.plot(frames, gta, linewidth=l, c=(255 / 255, 181 / 255, 112 / 255), alpha=a, label='Target A GT', zorder=2)

plt.plot(frames, gtb, linewidth=l, c=(126 / 255, 166 / 255, 224 / 255), alpha=a, label='Target B GT', zorder=2)

plt.scatter(frames, preda, color=(255 / 255, 181 / 255, 112 / 255), edgecolors='k', label='Target A Detection',
            zorder=3)
plt.scatter(frames, predb, color=(126 / 255, 166 / 255, 224 / 255), edgecolors='k', label='Target B Detection',
            zorder=3)

for i in range(0, T+1):
    if i % 10 == 0:
        plt.axvline(i, c='k', alpha=0.1, zorder=1)

plt.legend(loc=1, prop={'size': 7})
plt.xlabel('frames')
plt.ylabel('pixels')
plt.title('Horizontal Coordinate Trajectory')
plt.show()
