function W = getSimWarp(dx, dy, alpha_deg, lambda)
% alpha given in degrees, as indicated
rot = deg2rad(alpha_deg);
W = lambda * [cos(rot) -sin(rot) dx;
                sin(rot) cos(rot) dy];
         