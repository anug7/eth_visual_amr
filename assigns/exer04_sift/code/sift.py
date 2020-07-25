"""
SIFT implementation based on ETH Computer Vision Course
"""

import math
import numpy as np
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


def create_DOGs(img, base_sigma=1.6, scales=3):
  """
  Creates images in stack
  """
  stack_img = np.zeros((img.shape + (scales + 2, )))
  # prev_img = apply_conv(img, (2 ** (0 / octave)) * base_sigma)
  kx = 2 * math.ceil(2 * base_sigma) + 1
  prev_img = cv2.GaussianBlur(img, (kx, kx), (2.0 ** (-1.0 / scales)) * base_sigma)
  for s in range(0, scales + 2):
    sigma = (2.0 ** (s / scales)) * base_sigma
    kx = 2 * math.ceil(2 * sigma) + 1
    # cur_img = apply_conv(img, (2 ** (s / octave)) * base_sigma)
    cur_img = cv2.GaussianBlur(img, (kx, kx), sigma)
    diff = cur_img - prev_img
    stack_img[:, :, s] = diff
    prev_img = np.array(cur_img)
  return stack_img


def create_octaves(img, no_of_octaves=5):
  """
  """
  octaves, op_img = [], img.copy()
  for _ in range(no_of_octaves):
    octv = create_DOGs(op_img)
    octaves.append(octv)
    op_img = op_img[::2, ::2]  # resize image
  return octaves


def find_keypoints(octaves, scales):
  """
  Finds keypoint in the scale space
  """
  kps = []
  import ipdb; ipdb.set_trace()
  for idx, octave in enumerate(octaves):
    for s in range(1, scales + 1):
      for w in range(1, octave.shape[1] - 1):
        for h in range(1, octave.shape[0] - 1):
          vol = octave[h - 1:h + 2, w - 1:w + 2, s - 1: s + 2]
          max_val = np.max(vol)
          if max_val >= 0.04 and max_val == vol[1, 1, 1]:
            kps.append([w, h, s, idx])
  return kps

