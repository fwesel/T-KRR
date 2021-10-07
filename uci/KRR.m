function [w,X] = KRR(X,y,lengthscale,lambda)
    K = SE(X,X,lengthscale);
    w = (K+lambda*eye(length(y)))\y;
end
