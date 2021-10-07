function W = CPLS(X, y, M, R,lambda, lengthscale, numberSweeps)
    [~, D] = size(X);
    W = cell(1,D);
    Matd = 1;
    reg = 1;
    for d = D:-1:1
        W{d} = randn(M,R);
        W{d} = W{d}/norm(W{d});
        reg = reg.*vecnorm(W{d},2,1).^2;
        Mati = features(X(:,d),M,lengthscale);
        Matd = (Mati*W{d}).*Matd;
    end
    itemax = numberSweeps*(2*(D-1))+1;
    for ite = 1:itemax
        loopind = mod(ite-1,2*(D-1))+1;
        if loopind <= D
            d = loopind;
        else
            d = 2*D-loopind;
        end
        Mati = features(X(:,d),M,lengthscale);
        reg = reg./vecnorm(W{d},2,1).^2;
        Matd = Matd./(Mati*W{d});  
        [CC,Cy] = dotkron(Mati,Matd,y);
        x = (CC+lambda*diag(kron(reg,ones(1,M))))\Cy;
        clear CC Cy
        W{d} = reshape(x,M,R);
        reg = reg.*vecnorm(W{d},2,1).^2;
        Matd = Matd.*(Mati*W{d});  
    end
end