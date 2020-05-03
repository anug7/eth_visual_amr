

import numpy as np


def apply_conv(array, filter_array):
    """
    Applies 2D conv
    @oaram: array: input image
    @param: filter_array: conv filter array
    """
    pad_size = filter_array.shape[0] - 2
    pad_array = np.pad(array, (1,), "constant", constant_values=(0))
    sub_mat_size = filter_array.shape

    sub_matrices_shape = tuple(np.subtract(array.shape, sub_mat_size) + 1) + sub_mat_size
    strides = array.strides + array.strides

    sub_matrices = np.lib.stride_tricks.as_strided(array, sub_matrices_shape, strides)
    
    m = np.einsum('ij, klij->kl', filter_array, sub_matrices)

    return m


def calc_cornerness(Ix, Iy, k=0.04):
    """
    """
    Ix2 = (Ix**2).astype('float')
    Iy2 = (Iy**2).astype('float')
    Ixy2 = (Ix * Iy)**2
    first, second = Ix2 + Iy2, np.sqrt(4*Ixy2 + (Ix2-Iy2)**2)
    l1 = 0.5 * (first + second)
    l2 = 0.5 * (first - second)
    
    return l1, l2


def calc_harris_scores(l1, l2,k=0.04):
    return l1 * l2 - (k * (l1+ l2)**2)


def calc_tomashi_score(l1, l2):
    return np.min(l1, l2)
