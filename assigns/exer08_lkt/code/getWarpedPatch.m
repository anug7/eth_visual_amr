function patch = getWarpedPatch(I, W, x_T, r_T)
% patch = getWarpedPatch(I, W, x_T, r_T);
% x_T is 1x2 and contains [x_T y_T] as defined in the statement. patch is
% (2*r_T+1)x(2*r_T+1) and arranged consistently with the input image I.
% Input:
%   I, image
%   W, 2x3 affine transformation matrix
%   x_T, 1x2 patch translation (x,y)
%   r_T, scalar, patch radius
% Output:
%   patch, s_T x s_T patch, s_T = 2*r_T+1

patch = zeros(2*r_T+1);
max_coords = fliplr(size(I));
WT = W';
inter = true;
for x = -r_T:r_T
    for y = -r_T:r_T
        op = x_T + [x y 1] * WT;
        if all(op < max_coords & op > [1 1])
            if inter
                p0 = floor(op);
                coefs = op - p0;
                a = coefs(1);
                b = coefs(2);
                pval = (1-b) * (...
                    (1-a) * I(p0(2), p0(1)) +...
                    a * I(p0(2), p0(1)+1))...
                    + b * (...
                    (1-a) * I(p0(2)+1, p0(1)) +...
                    a * I(p0(2)+1, p0(1)+1));
                patch(y + r_T + 1, x + r_T + 1) = ...
                    pval;
            else
                patch(y + r_T + 1, x + r_T + 1) = ...
                    I(int16(op(2)), int16(op(1)));
            end
        end
    end
end
