import numpy as np

import cv2


def apply_conv(array, filter_array):
    """
    Applies 2D conv
    @oaram: array: input image
    @param: filter_array: conv filter array
    """
    pad_size = filter_array.shape[0] - 2
    pad_array = np.pad(array, (pad_size,), "constant", constant_values=(0))
    sub_mat_size = filter_array.shape

    sub_matrices_shape = tuple(np.subtract(array.shape, sub_mat_size) + 1) + sub_mat_size
    strides = array.strides + array.strides

    sub_matrices = np.lib.stride_tricks.as_strided(array, sub_matrices_shape, strides)
    
    m = np.einsum('ij, klij->kl', filter_array, sub_matrices)

    return m


def calc_cornerness(Ix, Iy, k=0.04):
    """
    """
    kern = np.asarray([[0.0625, 0.125 , 0.0625],
                       [0.125 , 0.25  , 0.125],
                       [0.0625, 0.125 , 0.0625]])
    Ix2 = (Ix**2)
    Iy2 = (Iy**2)
    
    sIx2 = apply_conv(Ix2, kern)
    sIy2 = apply_conv(Iy2, kern)
    
    Ixy = (Ix * Iy)
    sIxy = apply_conv(Ixy, kern)
    sIxy2 = sIxy ** 2 
    
    first, second = sIx2 + sIy2, np.sqrt(4*sIxy2 + (sIx2-sIy2)**2)
    l1 = 0.5 * (first + second)
    l2 = 0.5 * (first - second)
    
    return l1, l2


def calc_harris_scores(l1, l2,k=0.04):
    return l1 * l2 - (k * (l1+ l2)**2)


def calc_tomashi_score(l1, l2):
    return np.min(l1, l2)


def non_maxima_suppression(scores, no_of_points=40, w=3):
    locs = []
    for i in range(40):
        x, y = np.where(scores == np.amax(scores))
        scores[x[0] - w: x[0] + w, y[0] - w: y[0] + w] = np.zeros((w*2, w*2))
        locs.append((x[0], y[0]))
    return locs


def create_keypoint_desp(img, locs, d=144):
    """
    Create keypoint description based in image intensity values
    """
    desp, w = [], int(np.sqrt(d) / 2.)
    for l in locs:
        op = img[l[0] - w: l[0] + w, l[1] - w: l[1] + w].flatten()
        op = np.pad(op, (d - op.shape[0],), "constant", constant_values=(0))
        desp.append(op)
    return np.asarray(desp).T


def draw_keypoints(img, points, color=(0, 0, 255)):
    for p in points:
        img = cv2.circle(img,  (p[0], p[1]), 2, color, 2)
    return img
