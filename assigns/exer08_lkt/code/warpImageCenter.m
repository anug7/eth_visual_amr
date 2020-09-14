function I = warpImageCenter(I_R, W, cen)

I = zeros(size(I_R), 'uint8');
[height, width] = size(I_R);
for h=1:height
    for w=1:width
        op = int16(W * [w - cen(1); h - cen(2); 1]);
        if op(1) + cen(1) <= width && op(1) + cen(1) >= 1 && op(2) + cen(2) <= height && op(2) + cen(2) >=1
            I(h, w) = I_R(op(2) + cen(2), op(1) + cen(1));
        end 
    end
end