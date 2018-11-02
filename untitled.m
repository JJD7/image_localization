close all
clear all

img = imread('map/keas1.png');
test_img = imread('onboard_images/frame001378.png');
test_img = imresize(test_img, 0.1);
F = fft2(test_img);
H = fft2(img);

G = zeros(size(img));
kx = 128;
ky = 72;

for i=1:length(G(1,:,1))-kx:kx
    for j=1:length(G(:,1,1))-ky:ky
        H_subset = conj(H(j:j+ky-1,i:i+kx-1,:));
        col = i+kx-1;
        row = j+ky-1;
        G(j:row,i:col,:) = F.*H_subset;
    end
end

G_space = ifft(G);
imshow(G)