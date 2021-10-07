function Z = RFFPredict(X,W,B)
    M = size(W,2);
    Z = sqrt(2/M)*cos(X*W+B);
end