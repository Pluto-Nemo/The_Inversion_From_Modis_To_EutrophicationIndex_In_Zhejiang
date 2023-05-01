# VERSION 1.0
# fit for bpnn, gnnwr, stpnn+swnn
# 
# Contents:
# Spatial: Eurc
# Temporal: Abs

##### source：

import pandas as pd
import numpy as np
from numba import njit

def Compute_Dismat(spatial, temporal, spatial_list, temporal_list):
    """
    Caculate the spatial dismat and the temporal dismat with Eurc_dis and abs_time
    With no Grids but relative
    Output: spatial_dismat, temporal_dismat
    """
    spatial_dismat = []
    temporal_dismat = []
    N = spatial_list.shape[0]
    for i in range(N):
        if i % (N//10) == 0 and i!=0: print("{}0%".format(i//(N//10)),end=" ", flush=True)
        # caculate the ith spatial distance vector
        spatial_dis_x = spatial[:,0] - spatial_list[i,0]
        spatial_dis_y = spatial[:,1] - spatial_list[i,1]
        spatial_dis = spatial_dis_x ** 2 + spatial_dis_y ** 2
        spatial_dis = np.sqrt(spatial_dis)
        # caculate the ith temporal distance vector
        temporal_dis = temporal - temporal_list[i]
        temporal_dis = np.abs(temporal_dis) / 1e11
        # append into the matrix
        spatial_dismat.append(spatial_dis)
        temporal_dismat.append(temporal_dis)
    spatial_dismat = np.array(spatial_dismat)
    temporal_dismat = np.array(temporal_dismat)
    return spatial_dismat, temporal_dismat

def Compute_Dismat_Statistics(Dismat):
    """
    compute the means and the stds of the Dismat
    Output: means, stds
    """
    N = len(Dismat)
    means = []
    stds = []
    for i in range(N):
        means.append(Dismat[i].mean())
        stds.append(Dismat[i].std())
    means = np.array(means)
    stds = np.array(stds)
    return means, stds

def Compute_Dis_Standerlize(Dismat, means, stds):
    """ 
    standerlize the dismat with means and stds
    Output: Dismat
    """
    N = len(Dismat)
    N_m = len(means)
    N_s = len(stds)
    if N != N_m or N != N_s:
        print("Dimensions are not the same!")
    for i in range(N):
        Dismat[i] = (Dismat[i] - means[i]) / stds[i] + 1.0
    return Dismat

@njit()
def To3D_Trans(spatial_dismat, temporal_dismat):
    """
    Merge the two dismat and transform them
    Output: Var(b,a,2)
    """
    a = spatial_dismat.shape[0]
    b = spatial_dismat.shape[1]
    a_t = spatial_dismat.shape[0]
    b_t = spatial_dismat.shape[1]
    if a != a_t or b != b_t:
        print("Length division!")
    var = np.empty((b,a,2), dtype=float)
    for i in range(b):
        for j in range(a):
            var[i][j][0] = spatial_dismat[j][i]
            var[i][j][1] = temporal_dismat[j][i]
    return var