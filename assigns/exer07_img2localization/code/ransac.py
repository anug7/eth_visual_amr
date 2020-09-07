import numpy as np
from numpy.random import rand
from numpy import polyval

import matplotlib.pyplot as plt

np.random.seed(2)

def ransac_polynomial(X, Y, max_noise, no_of_points=3, iters=200):
  """
  RANSAC for polynomial fit with know max noise in data
  """
  end, cur_guess_idx, cur_guess_cnt = X.shape[0], 0, 0
  hist_inliers = []
  for i in range(iters):
    full_idcs = np.arange(end)
    np.random.shuffle(full_idcs)
    idcs1 = full_idcs[:no_of_points]
    idcs2 = full_idcs[no_of_points:]
    sampx = X[idcs1]
    sampy = Y[idcs1]
    testx = X[idcs2]
    testy = Y[idcs2]
    coefs = np.polyfit(sampx, sampy, 2)
    fwd = np.polyval(coefs, testx)
    vals = abs(fwd - testy)
    inliers = vals <= max_noise
    no_of_inliers = np.sum(inliers)
    if no_of_inliers >= cur_guess_cnt:
      hist_inliers.append({'coefs': coefs,
                              'inliers': [testx[inliers], testy[inliers]]})
      cur_guess_cnt = no_of_inliers
  inx = hist_inliers[-1]['inliers'][0]
  iny = hist_inliers[-1]['inliers'][1]
  ncoefs = np.polyfit(inx, iny, 2)
  return ncoefs

num_inliers = 20
num_outliers = 10
noise_ratio = 0.1
poly = rand(3)  # random second-order polynomial
extremum = -poly[1]/(2*poly[0])
xstart = extremum - 0.5
lowest = polyval(poly, extremum)
highest = polyval(poly, xstart)
xspan = 1
yspan = highest - lowest
max_noise = noise_ratio * yspan
x = rand(num_inliers) + xstart
y = polyval(poly, x)
y = y + (rand(y.shape[0])-.5) * 2 * max_noise

x_outlier = rand(num_outliers) + xstart
y_outlier = rand(num_outliers) * yspan + lowest

X = np.hstack((x, x_outlier))
Y = np.hstack((y, y_outlier))

plt.scatter(X, Y, marker="+")
ncoefs = ransac_polynomial(X, Y, max_noise)
plt.plot(np.sort(x), np.polyval(ncoefs, np.sort(x)))
plt.show()

print("op")
