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

    sub_matrices_shape = tuple(np.subtract(pad_array.shape, sub_mat_size) + 1) + sub_mat_size
    strides = pad_array.strides + pad_array.strides

    sub_matrices = np.lib.stride_tricks.as_strided(pad_array, sub_matrices_shape, strides)

    m = np.einsum('ij, klij->kl', filter_array, sub_matrices)

    return m


def calc_cornerness(Ix, Iy, k=0.04):
    """
    """
    kern = np.asarray([[0.0625, 0.125, 0.0625],
                       [0.125,  0.25,  0.125],
                       [0.0625, 0.125, 0.0625]])
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

    return (l1, l2)


def calc_harris_scores(l1, l2,k=0.04):
    return l1 * l2 - (k * (l1+ l2)**2)


def calc_tomashi_score(l1, l2):
    return np.min(l1, l2)


def non_maxima_suppression(scores, no_of_points=40, w=3):
    locs, shape = [], scores.shape
    for i in range(no_of_points):
        x, y = np.where(scores == np.amax(scores))
        x_start, x_end = max(0, x[0] - w), min(shape[0], x[0] + w)
        y_start, y_end = max(0, y[0] - w), min(shape[1], y[0] + w)
        size = (x_end - x_start, y_end - y_start)
        try:
            scores[x_start: x_end, y_start: y_end] = np.zeros(size)
        except:
            import ipdb; ipdb.set_trace()
        locs.append((x[0], y[0]))
    return locs


def create_keypoint_desp(img, locs, d=144):
    """
    Create keypoint description based in image intensity values
    """
    shape = img.shape
    desp, w = [], int(np.sqrt(d) / 2.)
    for l in locs:
        x_start, x_end = max(0, l[0] - w), min(shape[0], l[0] + w)
        y_start, y_end = max(0, l[1] - w), min(shape[1], l[1] + w)
        op = img[x_start: x_end, y_start: y_end].flatten()
        if op.shape[0] < d:
            op = np.append(op, np.asarray([0] * (d - op.shape[0])))
            if op.shape[0] < d:
                import ipdb; ipdb.set_trace()
        desp.append(op)
    return np.asarray(desp)


def match_desp(_trains, queries, lbda=10):
    """
    """
    trains = np.copy(_trains).astype('float')
    op = np.zeros(len(queries)).astype('int16')
    dists = np.ones(len(queries)) * np.NaN
    qry_idcs = np.arange(0, len(queries))
    for idx, qry in enumerate(queries):
        tmp = np.sqrt(np.sum((trains - qry)**2, axis=1))
        min_idx = np.nanargmin(tmp)
        dists[idx] = tmp[min_idx]
        op[idx] = min_idx
    min_dist = max(50, np.min(dists))
    print("Min dist: {} & counts: {}".format(min_dist, len(dists)))
    tidx = dists < (lbda * min_dist)
    dists = dists[tidx]
    op = op[tidx]
    qry_idcs = qry_idcs[tidx]
    uniqs, ucnts = np.unique(op, return_counts=True)
    dups = uniqs[ucnts > 1]
    for dup in dups:
        reps = np.where(op==dup)[0]
        midx = np.argmin(dists[reps])
        reps = np.delete(reps, midx)
        for rep in reps:
            op[rep] = -1
    filt_idcs = op > -1
    return qry_idcs[filt_idcs].tolist(), op[filt_idcs].tolist()


def draw_keypoints(img, points, color=(0, 0, 255)):
    for p in points:
        img = cv2.circle(img,  (p[1], p[0]), 2, color, 2)
    return img


def draw_matching_points(img, matches, train_locs, query_locs):
    """
    Draw matching keypoints between training and query images
    """
    for qry, trn in zip(matches[0], matches[1]):
        try:
            p2 = query_locs[qry]
            p1 = train_locs[trn]
            img = cv2.circle(img, (p1[1], p1[0]), 2, (0, 0, 255), 2)
            img = cv2.circle(img, (p2[1], p2[0]), 2, (0, 255, 0), 2)
            img = cv2.line(img, (p1[1], p1[0]), (p2[1], p2[0]), (255, 0, 0))
        except:
            import ipdb; ipdb.set_trace()
    return img


no_of_points = 100

img_path_fmt = '../data/{}.png'
img = cv2.imread(img_path_fmt.format(str(0).zfill(6)), 0)
dx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
dy = dx.T

Ix = apply_conv(img, dx)
Iy = apply_conv(img, dy)

l1, l2 = calc_cornerness(Ix, Iy)
haris = calc_harris_scores(l1, l2)
train_locs = non_maxima_suppression(haris, no_of_points=no_of_points)
train_desp = create_keypoint_desp(img, train_locs)
prev_img = img
cv2.namedWindow('t', 0)
#cv2.namedWindow('train', 0)
#cv2.namedWindow('query', 0)
wrtr = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 2, (1241, 376))

for idx in range(1, 200):
    img = cv2.imread(img_path_fmt.format(str(idx).zfill(6)), 0)

    Ix = apply_conv(img, dx)
    Iy = apply_conv(img, dy)

    l1, l2 = calc_cornerness(Ix, Iy)
    haris = calc_harris_scores(l1, l2)

    query_locs = non_maxima_suppression(haris, no_of_points=no_of_points)
    query_desp = create_keypoint_desp(img, query_locs)

    matches = match_desp(train_desp, query_desp, lbda=5.0)
    if matches:
        op_img = draw_matching_points(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), matches, train_locs, query_locs)
        cv2.imshow('t', op_img)
        #cv2.imshow('train', draw_keypoints(cv2.cvtColor(prev_img, cv2.COLOR_GRAY2RGB), train_locs))
        #cv2.imshow('query', draw_keypoints(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), query_locs))
        key = cv2.waitKey(10)
        if key == 27:
            break
    wrtr.write(op_img)
    train_locs = list(query_locs)
    train_desp = np.asarray(query_desp)
    prev_img = img
