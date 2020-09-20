function I = warpImageCenter(I_R, W, cen)

I = zeros(size(I_R), 'uint8');
I_R = padarray(I_R, [1 1], 'replicate', 'post');
[height, width] = size(I);
[hr, wr] = size(I_R);
for h=1:height
    for w=1:width
        op = (W * [w - cen(1); h - cen(2); 1]) + cen';
        p1 = ceil(op);
        coef = double(op - p1);
        if p1(1) < wr && p1(1) >= 1 && p1(2) < hr && p1(2) >=1
            matval = [I_R(p1(2), p1(1)) I_R(p1(2) + 1, p1(1)); I_R(p1(2), p1(1) + 1) I_R(p1(2) + 1, p1(1) + 1)];
            I(h, w) = [1 - coef(1) coef(1)] * double(matval)  * [1 - coef(2); coef(2)];
            %I(h, w) = I_R(op(2) + cen(2), op(1) + cen(1));
        end 
    end
end