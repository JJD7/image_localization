close all
clear all

img = rgb2gray(imread('map/keas1.png'));
test_img = rgb2gray(imread('onboard_images/frame001648.png'));
imshow(test_img)
test_img = imresize(test_img,0.15);


%rotate test image by yaw angle
test_img = imrotate(test_img,-30);

u = [1 0 -1]';
v = [1 2 1];
[tMag, tGir] = imgradient(test_img,'Prewitt');
[mMag, mDir] = imgradient(img,'Prewitt');

nMap = mMag-mean(mean(mMag));
nTest = tMag-mean(mean(tMag));

crr = xcorr2(nMap,nTest);

[ssr,snd] = max(crr(:));
[ij,ji] = ind2sub(size(crr),snd);

figure
plot(crr(:))
title('Cross-Correlation')
hold on
plot(snd,ssr,'or')
hold off
text(snd*1.05,ssr,'Maximum')

img(ij:-1:ij-size(test_img,1)+1,ji:-1:ji-size(test_img,2)+1) = rot90(test_img,2);

figure
imagesc(img)
axis image off
colormap gray
title('Reconstructed')