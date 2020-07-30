from heapq import heapify, heappush
import numpy as np
import cv2


limg = cv2.imread("../data/left/000000.png", 0)
rimg = cv2.imread("../data/right/000000.png", 0)

# Given by the KITTI dataset:
baseline = 0.54

# Carefully tuned by the TAs:
patch_radius = 5
min_disp, max_disp = 5, 50
disparity = np.zeros_like(limg)
for x in range(patch_radius + max_disp + 1, limg.shape[1] - patch_radius - 1):
    for y in range(patch_radius + 1, limg.shape[0] - patch_radius - 1):
        lblock = limg[y - patch_radius - 1: y + patch_radius,
                      x - patch_radius - 1: x + patch_radius]
        min_val, values, val_idx = 10000000, [], 0
        heapify(values)
        for d in range(min_disp, max_disp + 1):
            rblock = rimg[y - patch_radius - 1: y + patch_radius,
                          x - patch_radius - 1 - d: x + patch_radius - d]
            val = np.sum((rblock - lblock)**2)
            heappush(values, val)
            if val < min_val:
                val_idx = d
                min_val = val
        deno = float(values[0]) if values[0] > 0 else 200.0
        fact = np.asarray(values[1:4]) / deno
        if fact[0] < 1.5 and fact[1] < 1.5 and fact[2] < 1.5:
            disparity[y, x] = 0
        else:
            if val_idx != min_disp  and val_idx != max_disp:
                disparity[y, x] = val_idx

op_img = cv2.applyColorMap(disparity * 5, cv2.COLORMAP_JET)
cv2.imshow("op", op_img)
cv2.waitKey(0)
