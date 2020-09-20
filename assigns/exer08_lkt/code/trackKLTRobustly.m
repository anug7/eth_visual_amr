function [del_kps, keep] = trackKLTRobustly(I_prev, I, keypoint, ...
    r_T, num_iters, lambda)
% I_prev: reference image, I: image to track point in, keypoint: point to 
% track, expressed as [x y]=[col row], r_T: radius of patch to track, 
% num_iters: amount of iterations to run, lambda: bidirectional error
% threshold; delta_keypoint: delta by which the keypoint has moved between 
% images, (2x1), keep: true if the point tracking has passed the
% bidirectional error test.
W = trackKLT(I_prev, I, keypoint, r_T, num_iters);
del_kps = W(:, end);
Wrev = trackKLT(I, I_prev, (keypoint'+del_kps)', r_T, ...
    num_iters);
dkpinv = Wrev(:, end);
keep = norm(del_kps + dkpinv) < lambda;