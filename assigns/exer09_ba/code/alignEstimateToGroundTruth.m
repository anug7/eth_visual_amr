function p_G_C = alignEstimateToGroundTruth(...
    pp_G_C, p_V_C)
% Returns the points of the estimated trajectory p_V_C transformed into the
% ground truth frame G. The similarity transform Sim_G_V is to be chosen
% such that it results in the lowest error between the aligned trajectory
% points p_G_C and the points of the ground truth trajectory pp_G_C. All
% matrices are 3xN.

x0 = [HomogMatrix2twist(eye(4));1];
error_function = @(x) alignmentError(x, pp_G_C, p_V_C);
options = optimset('Display','MaxIter');
x = lsqnonlin(error_function, x0, [], [], options);
T = twist2HomogMatrix(x(1:6));
s = x(7);
R = T(1:3,1:3);
t = T(1:3,4);
p_G_C = s * R * p_V_C + t;
end


function error = alignmentError(x, pp_G_C, p_V_C)
    T = twist2HomogMatrix(x(1:6));
    s = x(7);
    R = T(1:3,1:3);
    t = T(1:3,4);
    pred = s * R * p_V_C + t;
    error = pp_G_C - pred; 
end