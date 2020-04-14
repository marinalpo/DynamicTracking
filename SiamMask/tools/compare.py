import numpy as np
mode_name = ['Single', 'Multi']

mode = 0
path = '/data/results/z_' + str(mode_name[mode]) + '.npy'
z_single = np.load(path)
# print('z_single:\n', z_single)

mode = 1
path = '/data/results/z_' + str(mode_name[mode]) + '.npy'
z_multi = np.load(path)
# print('z_multi:\n', z_multi)

equal = np.array_equal(z_single, z_multi)
equal2 = z_single - z_multi
print('Are equal:', equal)
print('Difference:', np.sum(equal2))