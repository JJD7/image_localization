close all
clear all

img = double(rgb2gray(imread('map/keas1.png')));
test_img = double(rgb2gray(imread('onboard_images/frame001378.png')));

test_img = imresize(test_img,0.1);
timg = test_img-mean(mean(test_img));


u = [1 0 -1]';
v = [1 2 1];
test_img = conv2(v,u,test_img,'same');


%creat image subset for testing
    x = 1000;
    X = 1300;
    szx = x:X;
    y = 1200;
    Y = 1500;
    szy = y:Y;
    Sect = img(szx,szy);
    
    pred_x = 600;
    pred_X = 1500;
    pred_win_x = pred_x:pred_X;
    pred_y = 600;
    pred_Y = 1500;
    pred_win_y = pred_y:pred_Y;


nimg = img-mean(mean(img));
[nimg_mag, nimg_grad] = imgradient(nimg,'Sobel');
nSec = nimg_mag(szx,szy);

predict = nimg_mag(pred_win_y,pred_win_x);

crr = xcorr2(predict,nSec);


[ssr,snd] = max(crr(:));
[ij,ji] = ind2sub(size(crr),snd);

figure
plot(crr(:))
title('Cross-Correlation')
hold on
plot(snd,ssr,'or')
hold off
text(snd*1.05,ssr,'Maximum')

%add overlay image on original to see if it matches
img(ij:-1:ij-size(Sect,1)+1,ji:-1:ji-size(Sect,2)+1) = rot90(Sect,2);

figure
imagesc(img)
axis image off
colormap gray
title('Reconstructed')
hold on
plot([y y Y Y y],[x X X x x],'r')
%plot([pred_y pred_y pred_Y pred_Y pred_y],[pred_x pred_X pred_X pred_x pred_x],'r')
hold off

%test_img = imresize(test_img, 0.1);
% F = fft2(test_img);
% H = fft2(img);
% 
% G = zeros(size(img));
% kx = length(test_img(1,:,:));
% ky = length(test_img(:,1,:));
% n = 0;
% 
% for i=1:kx:3500-kx
%     for j=1:ky:2560-ky
%         H_subset = conj(H(j:j+ky-1,i:i+kx-1,:));
%         col = i+kx-1;
%         row = j+ky-1;
%         G(j:row,i:col,:) = F.*H_subset;
%     end
% end
% 
% %G_space = ifft(G);
% 
% F2 = log(abs(G));
% imshow(F2)
% colormap(jet);