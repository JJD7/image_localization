close all
clear all

img = imread('map/keas1.png');
test_img = imread('onboard_images/frame001378.png');
test_img = imresize(test_img, 0.1);
F = fft2(test_img);
H = conj(fft2(img));

G = zeros(size(img));
kx = 128;
ky = 72;

for i=1:length(G(1,:,1))-kx
    for j=1:length(G(:,1,1))-ky
        H_subset = H(j:ky,i:kx,:);
        col = i+kx-1;
        row = j+ky-1;
        G_test = F.*H_subset;
    end
end

G_space = ifft2(G);
imshow(G)