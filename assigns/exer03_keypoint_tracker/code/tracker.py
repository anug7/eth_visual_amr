import numpy as np
import cv2

from scipy.signal import convolve2d


def calc_cornerness(Ix, Iy, patch_size=9):
  """
  """
  kern = np.ones((patch_size, patch_size)).astype('float') / (patch_size ** 2)
  Ix2 = (Ix**2)
  Iy2 = (Iy**2)
  Ixy = (Ix * Iy)

  sIx2 = convolve2d(Ix2, kern, "valid")
  sIy2 = convolve2d(Iy2, kern, "valid")

  sIxy = convolve2d(Ixy, kern, "valid")
  sIxy2 = sIxy ** 2

  det_muv = sIx2 * sIy2 - sIxy2
  trace2muv = (sIx2 + sIy2)**2
  return (det_muv, trace2muv)


def calc_harris_scores(l1, l2, patch_size=9, k=0.04):
  scores = l1 - k * l2
  scores[scores < 0] = 0
  pad_width = int((patch_size + 1) / 2)
  scores = np.pad(scores, (pad_width, pad_width), mode="constant",
                  constant_values=(0, 0))
  return scores


def calc_tomashi_score(l1, l2):
  return np.min(l1, l2)


def non_maxima_suppression(scores, no_of_points=40, r=9):
  locs = np.zeros((no_of_points, 2)).astype('int')
  tscores = np.pad(scores, (r, r), mode="constant", constant_values=(0, 0))
  for i in range(no_of_points):
    kp = np.unravel_index(tscores.argmax(), tscores.shape)
    # Subtract offset added by padding
    l = (np.array([kp[0], kp[1]]) - r).astype('int')
    locs[i, :] = l
    tscores[kp[0] - r:kp[0] + r + 1, kp[1] - r:kp[1] + r + 1] = 0.
  return locs


def create_keypoint_desp(img, locs, r=9):
  """
  Create keypoint description based in image intensity values
  """
  timg = np.pad(img, (r, r), mode="constant", constant_values=(0, 0))
  desp = np.zeros((locs.shape[0], (2 * r + 1)**2)) 
  for i, l in enumerate(locs):
    l[0], l[1] = l[0] + r, l[1] + r
    x_start, x_end = int(l[0]) - r, int(l[0]) + r + 1
    y_start, y_end = int(l[1]) - r, int(l[1]) + r + 1
    op = timg[x_start: x_end, y_start: y_end].flatten()
    desp[i, :] = op
  return desp


def match_desp(_trains, queries, lbda=10):
  """
  """
  trains = np.copy(_trains)
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
    img = cv2.circle(img,  (int(p[1]), int(p[0])), 2, color, 2)
  return img


def draw_matching_points(img, matches, train_locs, query_locs):
  """
  Draw matching keypoints between training and query images
  """
  for qry, trn in zip(matches[0], matches[1]):
    try:
      p2 = query_locs[qry]
      p1 = train_locs[trn]
      img = cv2.circle(img, (int(p1[1]), int(p1[0])), 2, (0, 0, 255), 2)
      img = cv2.circle(img, (int(p2[1]), int(p2[0])), 2, (0, 255, 0), 2)
      img = cv2.line(img, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), (255, 0, 0))
    except:
      import ipdb
      ipdb.set_trace()
  return img


def harris(qimg, corner_patch_size, kappa):
  """
  Find harris score for image
  """
  dx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  dy = dx.T
  Ix = convolve2d(qimg, dx, "valid")
  Iy = convolve2d(qimg, dy, "valid")

  l1, l2 = calc_cornerness(Ix, Iy, patch_size=corner_patch_size)
  harris_scores = calc_harris_scores(l1, l2, corner_patch_size, k=kappa)
  return harris_scores


def select_keypoints(harris_scores, no_of_kps=1000, nmx_radius=8):
  """
  Picks keypoints from haris score with Non-Maxima suppression
  """
  query_locs = non_maxima_suppression(harris_scores,
                                      no_of_points=no_of_kps,
                                      r=nmx_radius)
  return query_locs


def describe_keypoints(qimg, kps, dsp_rad):
  """
  Describes keypoint from image intensity values
  """
  desp = create_keypoint_desp(qimg, kps, r=dsp_rad)
  return desp

def match_descriptors(query, train, match_lambda):
  """
  Matches two descriptors and return the matches idcs in 
  query and train locs. 1-1 mapping
  """
  qry, trn = match_desp(train, query, lbda=match_lambda)
  return qry, trn


if __name__ == "__main__":
  no_of_points = 500
  
  img_path_fmt = '../data/{}.png'
  img = cv2.imread(img_path_fmt.format(str(0).zfill(6)), 0)
  dx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  dy = dx.T
  
  # Ix = apply_conv(img, dx)
  # Iy = apply_conv(img, dy)
  Ix = convolve2d(img, dx, mode="valid")
  Iy = convolve2d(img, dy, mode="valid")
  
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
  
    haris = harris(img, 8, 0.04)
 
    query_locs = select_keypoints(haris, no_of_kps=no_of_points, nmx_radius=8)
    query_desp = create_keypoint_desp(img, query_locs)
  
    #matches = match_desp(train_desp, query_desp, lbda=5.0)
    matches = match_descriptors(query_desp, train_desp, match_lambda=5.5)
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
