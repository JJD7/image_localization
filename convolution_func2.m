close all
clear all

% A = zeros(10);
% A(3:7,3:7) = ones(5);
% grayin = A;
% img=double(grayin);

input = imread('map/keas1.png');
grayin = rgb2gray(input);
img = double(grayin);



Gx = [1 0 -1;...'map/keas1.png'
    2 0 -2;...
    1 0 -1];
Gy = [1 2 1;...
    0 0 0;...
    -1 -2 -1];

% test = conv2(Gx,A);
% mesh(test)

% %flip kernals
% Gx = fliplr(Gx);
% Gy = flipud(Gy);

%expand matrix with border of zeros
A = zeros(size(img,1)+2,size(img,2)+2);
a = 2;
b = size(img,1)+1;
c = size(img,2)+1;
A(a:b,a:c) = img;

%x-gradients. Replaces: tmpX = conv2(double(img), Gx)
for i=1:size(A,2)-2
    for j=1:size(A,1)-2
        tmpX(j,i) = ...
                A(j,i)*Gx(1,1)+A(j+1,i)*Gx(2,1)+A(j+2,i)*Gx(3,1)+...
                A(j,i+1)*Gx(1,2)+A(j+1,i+1)*Gx(2,2)+A(j+2,i+1)*Gx(3,2)+...
                A(j,i+2)*Gx(1,3)+A(j+1,i+2)*Gx(2,3)+A(j+2,i+2)*Gx(3,3);
    end
end

%y-gradients; replaces: tmpY = conv2(double(img), Gy)
for i=1:size(A,2)-2
    for j=1:size(A,1)-2
        tmpY(j,i) = ...
                A(j,i)*Gy(1,1)+A(j+1,i)*Gy(2,1)+A(j+2,i)*Gy(3,1)+...
                A(j,i+1)*Gy(1,2)+A(j+1,i+1)*Gy(2,2)+A(j+2,i+1)*Gy(3,2)+...
                A(j,i+2)*Gy(1,3)+A(j+1,i+2)*Gy(2,3)+A(j+2,i+2)*Gy(3,3);
    end
end

figure
test1 = conv2(Gx,A);
test2 = conv2(Gy,A);
mesh(sqrt(test1.^2+test2.^2))

tmp = sqrt(tmpX.*tmpX + tmpY.*tmpY);
[rT cT] = size(tmp);
tmp = min(tmp, 255);
edge = uint8(tmp(2:rT-1, 2:cT-1));

figure
mesh(tmp)
