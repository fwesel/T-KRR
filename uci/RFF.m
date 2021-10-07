function [ZZ,ZY,W,B] = RFF(X,Y,M,lengthscale)
    batchSize = 100;
    [N,D] = size(X);
    W = normrnd(0,1/lengthscale,D,M);
    B = 2*pi*rand(1,M);
    ZZ = zeros(M,M);
    ZY = zeros(M,1);
    for n = 1:batchSize:N
        idx = min(n+batchSize-1,N);
        temp = sqrt(2/M)*cos(X(n:idx,:)*W+B);
        ZZ = ZZ+temp'*temp;
        ZY = ZY+temp'*Y(n:idx);
    end
end