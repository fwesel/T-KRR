function K = SE(X1,X2,lengthscale)
    K = exp(-0.5*pdist2(X1,X2).^2/(lengthscale^2));
end