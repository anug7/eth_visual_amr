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
  history_inliers = []
  for i in range(iters):
    idcs = np.random.randint(0, end, size=no_of_points)
    sampx, sampy = X[idcs], Y[idcs]
    coefs = np.polyfit(sampx, sampy, 2)
    fwd = np.polyval(coefs, X)
    vals = abs(fwd - Y)
    no_of_inliers = np.sum(vals <= max_noise)
    if no_of_inliers >= cur_guess_cnt:
      history_inliers.append({'coefs': coefs,
          'inliers': [X[vals >= max_noise], Y[vals>=max_noise]]})
      cur_guess_cnt = no_of_inliers
      plt.scatter(X, fwd)
  return history_inliers

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

hist_inliers = ransac_polynomial(X, Y, max_noise)
