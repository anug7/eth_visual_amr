import numpy as np


def get_sim_wrap(dx, dy, alpha, lbd):
  """
  Returns affine transformation matrix based on
  given parameters
  """
  rot = np.deg2rad(alpha)
  rmat = np.asarray([[np.cos(rot), -np.sin(rot), dy],
                     [np.sin(rot), np.cos(rot), dx]])
  return lbd * rmat


def warp_img_orig(img, warp):
  """
  Warps image based on warp matrix obtained
  """
  wimg = np.zeros_like(img)
  height, width = img.shape[:2]
  for h in range(img.shape[0]): # height
    for w in range(img.shape[1]): # width
      op = warp.dot(np.asarray([[w], [h], [1]])).astype('int')
      if op[0] < width and op[0] >= 0 \
        and op[1] < height and op[1] >= 0:
        wimg[h, w, :] = img[op[1], op[0], :]
  return wimg


def warp_img_center(img, warp, wcen=[]):
  """
  Warps image based on warp matrix obtained
  """
  wimg = np.zeros_like(img)
  height, width = img.shape[:2]
  if wcen:
      cen = wcen[::-1]
  else:
    cen = [int((height + 1) / 2), int((width + 1) / 2)]
  for h in range(img.shape[0]): # height
    for w in range(img.shape[1]): # width
      op = warp.dot(np.asarray([[w - cen[1]], [h - cen[0]], [1]])).astype('int')
      op = op.flatten()
      if op[0] + cen[1] < width and op[0] + cen[1] >= 0 \
        and op[1] + cen[0] < height and op[1] + cen[0] >= 0:
        wimg[h, w, :] = img[op[1] + cen[0], op[0] + cen[1], :]
  return wimg

def get_warped_patch(img, warp, loc, patch_radius):
  """
  Gets patch from the image
  """
  op_img = warp_img_center(img, warp, wcen=loc)
  patch = op_img[loc[1] - patch_radius:loc[1] + patch_radius + 1,
                 loc[0] - patch_radius:loc[0] + patch_radius + 1]
  return patch


def get_warp_brute_force(img, timg, patch_radius, search_radius):
  """
  Recover patch with simple brute force matching
  @param: img[np.array]: input image
  @param: timg[np.array]: image in which patch has to recovered
  @param: patch_radius[int]: patch radius
  @param: search_radius[int]: search length along x & y axes
  @param: loc[[int, int]]: center of patch in input image
  """
