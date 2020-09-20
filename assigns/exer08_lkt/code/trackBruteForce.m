function [dx, ssds] = trackBruteForce(I_R, I, x_T, r_T, r_D)
% I_R: reference image, I: image to track point in, x_T: point to track,
% expressed as [x y]=[col row], r_T: radius of patch to track, r_D: radius
% of patch to search dx within; dx: translation that best explains where
% x_T is in image I, ssds: SSDs for all values of dx within the patch
% defined by center x_T and radius r_D.
template = I_R(x_T(2) - r_T:x_T(2) + r_T, x_T(1) - r_T:x_T(1) + r_T);
ssds = zeros(2* r_D + 1, r_D*2 + 1);
for i=-r_D:r_D
    for j=-r_D:r_D
        tmp = I(x_T(2) + j - r_T:x_T(2) + j + r_T, x_T(1) + i - r_T:x_T(1) + i + r_T);
        ssds(i + r_D + 1, j + r_D + 1) = sum(sum((template - tmp).^2));
    end
end
[M, dx] = minmat(ssds);

% Brute force with wrapped
%W = getSimWarp(i, j, 0, 1);
%patch = getWarpedPatch(I, W, x_T, r_T);
%ssds(i + r_D +1, j + r_D + 1) = sum(sum((template-patch).^2));