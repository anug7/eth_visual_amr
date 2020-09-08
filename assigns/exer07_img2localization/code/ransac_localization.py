import sys

import numpy as np
import cv2

sys.path.insert(0, "../../exer02_pnp/code/")
sys.path.insert(0, "../../exer03_keypoint_tracker/code/")

import tracker as tk

harris_patch_size = 9
harris_kappa = 0.08
non_maxima_supp_radius = 8
descriptor_radius = 9
match_lambda = 5

no_of_keypoints = 1000

kmat = np.loadtxt("../data/K.txt")
db_keypoints = np.loadtxt("../data/keypoints.txt")
p_w_landmarks = np.loadtxt("../data/p_W_landmarks.txt")

dimg = cv2.imread("../data/000000.png", 0)
qimg = cv2.imread("../data/000001.png", 0)

qharris = tk.harris(qimg, harris_patch_size, harris_kappa)
qkps = tk.select_keypoints(qharris, no_of_keypoints, non_maxima_supp_radius)
qdesp = tk.describe_keypoints(qimg, qkps, descriptor_radius)

ddesp = tk.describe_keypoints(dimg, db_keypoints, descriptor_radius)

matches = tk.match_descriptors(qdesp, ddesp, match_lambda)

op_img = tk.draw_matching_points(cv2.cvtColor(qimg, cv2.COLOR_GRAY2BGR), matches, db_keypoints, qkps)

cv2.imshow("op", op_img)
cv2.waitKey(0)
