import sys

import numpy as np
import cv2

sys.path.insert(0, "../../exer01_virtual_cube/code")
sys.path.insert(0, "../../exer02_pnp/code/")
sys.path.insert(0, "../../exer03_keypoint_tracker/code/")

import tracker as tk
import pnp as pnp
import draw_cube as dc

np.random.seed(42)

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
qdesp = tk.describe_keypoints(qimg, qkps, descriptor_radius * 2)

ddesp = tk.describe_keypoints(dimg, db_keypoints, (descriptor_radius * 2))

matches = tk.match_descriptors(qdesp, ddesp, match_lambda)

mat_query, mat_train = matches
idcs = np.arange(db_keypoints.shape[0]).tolist()
rem_train = [idx for idx in idcs if idx not in mat_train]

op_img = tk.draw_matching_points(cv2.cvtColor(qimg, cv2.COLOR_GRAY2BGR), matches, db_keypoints, qkps)
op_img = tk.draw_keypoints(op_img, db_keypoints[rem_train])

def ransace_dlt(dbkps, dbObj, kmat, msample=6):
  """
  Ransac for DLT to remove outliers
  """
  dbkps = np.hstack((dbkps, np.ones((dbkps.shape[0], 1))))
  dbObj = np.hstack((dbObj, np.ones((dbObj.shape[0], 1))))
  for _ in range(2000):
    idcs = np.arange(dbkps.shape[0])
    np.random.shuffle(idcs)
    model_idcs = idcs[:msample]
    test_idcs = idcs[msample:]
    samp_ipts = dbkps[model_idcs]
    samp_wpts = dbObj[model_idcs]
    # samp_ipts = np.linalg.inv(kmat).dot(samp_ipts.T).T
    aug = pnp.estimatePoseDLT(samp_ipts,
                              samp_wpts,
                              kmat)
    test_wpts = dbObj[test_idcs]
    reproj_pts = dc.projectPoints_mat(kmat, aug, test_wpts)
    test_ipts = dbkps[test_idcs]
    proj_err = np.sum((reproj_pts - test_ipts[:, :2])**2, axis=1)
    if np.sum(proj_err <= 100.0) >=6:
      print("Min")
    

ransace_dlt(np.asarray(qkps)[mat_query], np.asarray(p_w_landmarks)[mat_train], kmat)

cv2.imshow("op", op_img)
cv2.waitKey(0)
