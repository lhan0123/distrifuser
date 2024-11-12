import numpy as np
import matplotlib.pyplot as plt


def show_patch_maps():
    patch_map = np.loadtxt('clusters/cluster_1/patch_map.txt', delimiter=',')
    plt.imshow(patch_map, cmap='hot')
    plt.axis('off')
    plt.savefig('clusters/cluster_0/patch_map.png')
    

show_patch_maps()