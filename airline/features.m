function Mati = features(X,M,lengthscale)
    X = (X+1/2)/2;
    w = 1:M;
    S = sqrt(2*pi)*lengthscale*exp(-(pi*w/2).^2*lengthscale^2/2);
    Mati = sinpi(X*w).*sqrt(S);
end