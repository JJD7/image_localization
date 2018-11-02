%convolution function for edge detection
close all

A = zeros(10);
A(3:7,3:7) = ones(5);
mesh(A)

u = [1 0 -1]';
v = [1 2 1];
ch =[];
Ch = [];
cv = [];
Cv = [];

%convolve the rows of A with vector u
for i=1:size(A,2)
    for j=1:size(A,1)
        if j-1==0  
            ch(j,i) = A(j,i)*u(2)+A(j+1,i)*u(1);
        elseif j==size(A,1)
            ch(j,i) = A(j-1,i)*u(3)+A(j,i)*u(2);
            else
            ch(j,i) = A(j,i)*u(2)+A(j-1,i)*u(3)+A(j+1,i)*u(1);
        end
    end
end

%convolve the rows of ch with vector v
for i=1:size(ch,2)
    for j=1:size(ch,1)
        if i-1==0  
            Ch(j,i) = ch(j,i)*v(2)+ch(1,i+1)*v(1);
        elseif i==size(ch,2)
            Ch(j,i) = ch(j,i-1)*v(3)+ch(j,i)*v(2);
            else
            Ch(j,i) = ch(j,i)*v(2)+ch(j,i-1)*v(3)+ch(j,i+1)*v(1);
        end
    end
end

figure
mesh(Ch)

%test convolution algorithm using matlabs functions
wif = conv2(v,u,A,'same');
figure
mesh(wif)

%convolve the rows of A with vector v
for i=1:size(A,2)
    for j=1:size(A,1)
        if i-1==0  
            cv(j,i) = A(j,i)*u(2)+A(1,i+1)*u(1);
        elseif i==size(A,2)
            cv(j,i) = A(j,i-1)*u(3)+A(j,i)*u(2);
            else
            cv(j,i) = A(j,i)*u(2)+A(j,i-1)*u(3)+A(j,i+1)*u(1);
        end
    end
end

%convolve the rows of cv with vector u
for i=1:size(cv,2)
    for j=1:size(cv,1)
        if j-1==0  
            Cv(j,i) = cv(j,i)*v(2)+cv(j+1,i)*v(1);
        elseif j==size(cv,1)
            Cv(j,i) = cv(j-1,i)*v(3)+cv(j,i)*v(2);
            else
            Cv(j,i) = cv(j,i)*v(2)+cv(j-1,i)*v(3)+cv(j+1,i)*v(1);
        end
    end
end

figure
mesh(Cv)

edge = sqrt(Ch.^2+Cv.^2);

figure
mesh(edge)