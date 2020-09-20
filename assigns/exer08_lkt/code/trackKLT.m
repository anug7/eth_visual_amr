function [W, p_hist] = trackKLT(I_R, I, x_T, r_T, num_iters)
% I_R: reference image, I: image to track point in, x_T: point to track,
% expressed as [x y]=[col row], r_T: radius of patch to track, num_iters:
% amount of iterations to run; W(2x3): final W estimate, p_hist 
% (6x(num_iters+1)): history of p estimates, including the initial
% (identity) estimate
filt = [0 0 0; 1 0 -1; 0 0 0];
W = getSimWarp(0, 0, 0, 1);
temp = getWarpedPatch(I_R, W, x_T, r_T);
i_r = temp(:);
p_hist = zeros(6, num_iters + 1);
p_hist(:, 1) = reshape(W, 1, []);
n = r_T * 2 + 1;
do_plot = false;
for iter=1:num_iters
    temp_new = getWarpedPatch(I, W, x_T, r_T + 1);
    dxI = conv2(temp_new, filt, 'valid');
    dyI = conv2(temp_new, filt', 'valid');
    op = zeros(n^2, 6);
    idx = 1;
    %update as err matrix ordering.. row wise ordering
    for i=-r_T:r_T
        for j=-r_T:r_T
            dx = dxI(r_T + j + 1, r_T + i + 1);
            dy = dyI(r_T + j + 1, r_T + i + 1);
            dw = [i 0 j  0 1 0; 0 i 0 j 0 1];
            op(idx, :) = [dx dy]* dw;
            idx = idx + 1;
        end
    end
    temp_new = temp_new(2:end -1, 2:end-1); 
    i_new = temp_new(:);
    H = op' * op;
    dW = H^-1  * op' * double(i_r - i_new);
    W = W + reshape(dW, [2, 3]);
    p_hist(:, iter + 1) = reshape(W, [6, 1]);
    if do_plot
        subplot(3, 1, 1);
        imagesc([temp_new temp (temp - temp_new)]);
        title('I(W(T)), I_R(T) and the difference')
        colorbar;
        axis equal;
        subplot(3, 1, 2);
        imagesc([dxI dyI]);
        title('warped gradients')
        colorbar;
        axis equal;
        subplot(3, 1, 3);
        grads = zeros(n, 6*n);
        for j = 1:6
            grads(:, (j-1)*n+1:j*n) = reshape(op(:, j), [n n]);
        end
        imagesc(grads);
        title('steepest descent images');
        colorbar;
        axis equal;
        pause(0.1)
    end
    if norm(dW) < 1e-3
        break
    end
end