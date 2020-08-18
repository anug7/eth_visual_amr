"""
SIFT implementation based on ETH Computer Vision Course
"""

import math
import numpy as np

from matplotlib import pyplot as plt
import cv2


def apply_conv(img, sigma):
  """
  Apply convulation on the input image
  """
  kx = cv2.getGaussianKernel(5, sigma)
  ky = kx
  filter_array = kx * ky.T

  pad_size = filter_array.shape[0] - 2
  pad_array = np.pad(img, (pad_size,), "constant", constant_values=(0))
  sub_mat_size = filter_array.shape

  sub_matrices_shape = tuple(np.subtract(pad_array.shape, sub_mat_size) + 1) + sub_mat_size
  strides = pad_array.strides + pad_array.strides

  sub_matrices = np.lib.stride_tricks.as_strided(pad_array, sub_matrices_shape, strides)

  m = np.einsum('ij, klij->kl', filter_array, sub_matrices)

  return m


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def compute_gradient(img_block):
  """
  imgradient MATLAB function equivalent
  """
  sobelx = cv2.Sobel(img_block, cv2.CV_64F, 1, 0)  # Find x and y gradients
  sobely = cv2.Sobel(img_block, cv2.CV_64F, 0, 1)

  # Find magnitude and angle
  magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
  angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
  return magnitude, angle


def compute_weighted_hist(ang, mag, bins):
  """
  Computers orientation histogram based on orientation and its corresponding
  gradient
  @param: ang[np.array]: orientation matrix
  @param: mag[np.array]: correspinding gradient matrix
  @param: bins[np.array]: no of bins the orientation histogram
  """
  hists = np.zeros_like(bins)
  for idx, abin in enumerate(bins[:-1]):
    idcs = np.where(np.logical_and(ang < bins[idx + 1], ang >= bins[idx]))
    if len(idcs) > 0:
      hists[idx] = np.sum(mag[idcs])
  idcs = ang == bins[-1]
  if len(idcs) > 0:
    hists[-1] = np.sum(mag[idcs])

  return hists


def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
  """
  Return True if the center element of the 3x3x3 input array is strictly greater
  than or less than all its neighbors, False otherwise
  """
  center_pixel_value = second_subimage[1, 1]
  if center_pixel_value > threshold:
    return np.all(center_pixel_value >= first_subimage) and \
           np.all(center_pixel_value >= third_subimage) and \
           np.all(center_pixel_value >= second_subimage[0, :]) and \
           np.all(center_pixel_value >= second_subimage[2, :]) and \
           center_pixel_value >= second_subimage[1, 0] and \
           center_pixel_value >= second_subimage[1, 2]
  return False


def create_DOGs(img, base_sigma=1.6, scales=3):
  """
  Creates images in stack
  """
  stack_img, blrd_imgs = np.zeros((img.shape + (scales + 2, ))), {}
  # prev_img = apply_conv(img, (2 ** (0 / octave)) * base_sigma)
  kx = 2 * math.ceil(2 * base_sigma) + 1
  prev_img = cv2.GaussianBlur(img, (kx, kx), (2.0 ** (-1.0 / scales))
                              * base_sigma)
  blrd_imgs[-1] = prev_img.copy()
  for s in range(0, scales + 2):
    blrd_imgs[s] = compute_gradient(prev_img)
    sigma = (2.0 ** (s / scales)) * base_sigma
    kx = 2 * math.ceil(2 * sigma) + 1
    # cur_img = apply_conv(img, (2 ** (s / octave)) * base_sigma)
    cur_img = cv2.GaussianBlur(img, (kx, kx), sigma)
    diff = np.abs(cur_img - prev_img)
    stack_img[:, :, s] = diff
    prev_img = np.array(cur_img)
  return stack_img, blrd_imgs


def create_octaves(img, no_of_octaves=5):
  """
  """
  octaves, op_img = [], img.copy()
  blrd_imgs = []
  for _ in range(no_of_octaves):
    octv, blrd_img = create_DOGs(op_img)
    octaves.append(octv)
    blrd_imgs.append(blrd_img)
    op_img = op_img[::2, ::2]  # resize image
  return octaves, blrd_imgs


def find_keypoints(octaves, scales):
  """
  Finds keypoint in the scale space
  """
  kps = []
  for idx, octave in enumerate(octaves):
    for s in range(1, scales + 1):
      # TODO: Handle w increment based on h also
      for w in range(1, octave.shape[1] - 1):
        for h in range(1, octave.shape[0] - 1):
          vol = octave[h - 1:h + 2, w - 1:w + 2, s - 1: s + 2]
          if isPixelAnExtremum(vol[:, :, 0], vol[:, :, 1], vol[:, :, 2], 0.04):
            kps.append([w, h, s, idx])
  return kps


def create_description(kps, blrd_imgs):
  """
  Create sift description for detected keypoints
  """
  angles = np.asarray(range(-180, 180, 45)).astype('float32')
  wg = matlab_style_gauss2D((16, 16), 1.5 * 16)
  kp_in_orig, kdesp = [], []
  for kp in kps:
    x, y, s, o = kp
    mag, ang = blrd_imgs[o][s - 1]
    h, w = mag.shape[:2]
    if x >= 8 and (x + 8) < w and y >= 8 and (y + 8) < h:
      bmag = mag[y - 8:y + 8, x - 8:x + 8]
      bang = ang[y - 8:y + 8, x - 8:x + 8]
      bmag = (bmag * wg).reshape((64, 4))
      bang = bang.reshape((64, 4))
      tdesp, r = np.zeros((128,)), 0
      for j in range(0, 16):
         _bmag = bmag[j * 4:(j * 4) + 4, :]
         _bang = bang[j * 4:(j * 4) + 4, :]
         desp = compute_weighted_hist(_bang, _bmag, angles)
         tdesp[r: r + 8], r = desp, r + 8
      kp_in_orig.append([x * 2**o, y * 2**o])
      kdesp.append(tdesp / np.linalg.norm(tdesp))
  return kp_in_orig, kdesp


def convert2OpenCV(desp):
  """
  """
  op = np.round(np.asarray(desp, "float32") * 512)
  op[op < 0] = 0
  op[op > 255] = 255

  return op


train = cv2.imread("../images/img_1.jpg", 0)
query = cv2.imread("../images/img_2.jpg", 0)

# train = cv2.imread("/home/guna007/tmp/tmp/PythonSIFT/box_in_scene.png", 0)
# query = cv2.imread("/home/guna007/tmp/tmp/PythonSIFT/box.png", 0)

train_op, train_gaus = create_octaves(train / 255.0)
query_op, query_gaus = create_octaves(query / 255.0)


train_kps = find_keypoints(train_op, 3)
query_kps = find_keypoints(query_op, 3)

del train_op
del query_op

train_keypoints, train_desp = create_description(train_kps, train_gaus)
query_keypoints, query_desp = create_description(query_kps, query_gaus)

del train_gaus, query_gaus

train_desp = convert2OpenCV(train_desp)
query_desp = convert2OpenCV(query_desp)

import ipdb; ipdb.set_trace()
# Initialize and use FLANN
FLANN_INDEX_KDTREE, MIN_MATCH_COUNT = 0, 5
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(query_desp, train_desp, k=2)
# matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([query_keypoints[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([train_keypoints[m.trainIdx] for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = query.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    train = cv2.polylines(train, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = query.shape
    h2, w2 = train.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = query
        newimg[:h2, w1:w1 + w2, i] = train

    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(query_keypoints[m.queryIdx][0]), int(query_keypoints[m.queryIdx][1] + hdif))
        pt2 = (int(train_keypoints[m.trainIdx][0] + w1), int(train_keypoints[m.trainIdx][1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    cv2.imwrite("output.jpeg", newimg)
    #plt.imshow(newimg)
    #plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

