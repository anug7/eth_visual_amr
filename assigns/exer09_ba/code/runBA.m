function hidden_state = runBA(hidden_state, observations, K)
% Update the hidden state, encoded as explained in the problem statement,
% with 20 bundle adjustment iterations.
use_pattern = true;
pattern = [];

if use_pattern
    % code for pattern
    N = observations(1);
    L = observations(2);
    num_patterns = size(hidden_state,1);
    num_error = (numel(observations) - 2 - N)/3;
    pattern_x = 2*num_error;
    index = 3;
    pattern = spalloc(pattern_x, num_patterns, pattern_x * 9);
    
    frame_ind = 1;
    for i = 1:N
        num_features = int64(observations(index));
        y_start_ind = index + 1;
        y_end_ind = 2*num_features + y_start_ind - 1 ;
        land_start_ind = y_end_ind + 1;
        land_end_ind = num_features + land_start_ind - 1;
        % get landmarks
        cur_land_ind = observations(land_start_ind: land_end_ind);
        
        % settting the poses to 1
        pose_start = ((i - 1) * 6) + 1;
        pose_end = pose_start + 6 - 1;
        pattern(frame_ind : frame_ind + 2*num_features - 1, pose_start:pose_end) = 1;
        for j = 1:num_features
            % setting landmarks to 1, last index
            land_index = cur_land_ind(j);
            start_land = (6*N) + (3*land_index) - 2;
            end_land = start_land + 3 - 1;
            pattern(frame_ind + (j - 1)*2 :frame_ind + (j - 1)*2 + 1  , start_land : end_land) = 1;
        end
        frame_ind = frame_ind + 2*num_features;
        index = land_end_ind + 1;
    end
    
    figure(10)
    spy(pattern)
end


x0 = hidden_state;
error_function = @(x) reprojection_error(x, observations, K);
options = optimset('Display','iter', 'parallel_local', true, 'MaxIter', 20);
if use_pattern
    options.JacobPattern = pattern;
    options.UseParallel = true;
end
hidden_state = lsqnonlin(error_function, x0, [], [], options);
end


function error = reprojection_error(x, observations, K)
    N = observations(1);
    L = observations(2);
    poses = reshape(x(1:6*N), 6, N);
    landmarks = reshape(x(6*N + 1 : end), 3, L);
    error = [];
    index = 3;
    
    for i = 1:N
        num_features = int64(observations(index));
        y_start_ind = index + 1;
        y_end_ind = 2*num_features + y_start_ind - 1 ;
        obs_px = reshape(observations(y_start_ind: y_end_ind), 2, num_features);
        obs_px = [obs_px(2,:);obs_px(1,:)];
        land_start_ind = y_end_ind + 1;
        land_end_ind = num_features + land_start_ind - 1;
 
        cur_land_ind = observations(land_start_ind: land_end_ind);
 
        cur_land = landmarks(:, cur_land_ind);
        cur_land_hom = [cur_land; ones(1, num_features)];

        cur_pose = twist2HomogMatrix(poses(:,i))^-1;
        fx = K * cur_pose(1:3,1:4) * cur_land_hom; 
        fx = fx ./ fx(3,:);
        fx = fx(1:2,:);
        
        index = land_end_ind + 1;
        
        %reprojection error       
        frame_error = fx - obs_px;
        error = [error, frame_error];
    end
end
