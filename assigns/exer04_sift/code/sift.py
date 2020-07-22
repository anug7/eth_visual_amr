"""
SIFT implementation based on ETH Computer Vision Course
"""

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


def create_octave(img, base_sigma=1.6, octave=3):
  """
  Creates images in stack
  """
  stack_img = np.zeros((img.shape + (octave + 2, )))
  # prev_img = apply_conv(img, (2 ** (0 / octave)) * base_sigma)
  prev_img = cv2.GaussianBlur(img, (11, 11), (2 ** (0 / octave)) * base_sigma)
  for s in range(0, octave + 2):
    # cur_img = apply_conv(img, (2 ** (s / octave)) * base_sigma)
    cur_img = cv2.GaussianBlur(img, (11, 11), (2 ** (s / octave)) * base_sigma)
    diff = cur_img - prev_img
    prev_img = cur_img
    stack_img[:, :, s] = diff
  return stack_img
