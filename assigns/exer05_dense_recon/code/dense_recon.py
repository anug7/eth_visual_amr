
import numpy as np
import cv2


limg = cv2.imread("../data/left/000000.png", 0)
rimg = cv2.imread("../data/right/000000.png", 0)

# Given by the KITTI dataset:
baseline = 0.54;

# Carefully tuned by the TAs:
patch_radius = 5
min_disp = 5
max_disp = 50
disparity = np.zeros_like(limg)
for x in range(patch_radius + max_disp + 1, limg.shape[1] - patch_radius - 1):
    for y in range(patch_radius + 1, limg.shape[0] - patch_radius - 1):
        lblock = limg[y - patch_radius - 1: y + patch_radius,
                      x - patch_radius - 1: x + patch_radius]
        val_idx, min_val = 0, 10000000
        for d in range(5, 50):
            rblock = rimg[y - patch_radius - 1: y + patch_radius,
                          x - patch_radius - 1 - d: x + patch_radius - d]
            val = np.sum((rblock - lblock)**2)
            if val < min_val:
                val_idx = d
                min_val = val
        disparity[y, x] = val_idx
