function I = warpImage(I_R, W)

I = zeros(size(I_R), 'uint8');
[height, width] = size(I_R);
for h=1:height
    for w=1:width
        op = int16(W * [w; h; 1]);
        if op(1) <= width && op(1) >= 1 && op(2) <= height && op(2) >=1
            I(h, w) = I_R(op(2), op(1));
        end 
    end
end
