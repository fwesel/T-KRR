function [W, loss, error] = CPLS(X, y, M, R ,lambda, lengthscale, numberSweeps)
    [~, D] = size(X);
    W = cell(1,D);
    Matd = 1;
    reg = 1;
    for d = D:-1:1
        W{d} = randn(M,R);
        W{d} = W{d}/norm(W{d});
        reg = reg.*(W{d}'*W{d});
        Mati = features(X(:,d),M,lengthscale);
        Matd = (Mati*W{d}).*Matd;
    end

    itemax = numberSweeps*(2*(D-1))+1;
    loss = zeros(itemax,1);  error = zeros(itemax,1);
    for ite = 1:itemax
        loopind = mod(ite-1,2*(D-1))+1;
        if loopind <= D
            d = loopind;
        else
            d = 2*D-loopind;
        end
        Mati = features(X(:,d),M,lengthscale);
        reg = reg./(W{d}'*W{d});
        Matd = Matd./(Mati*W{d});
        C = dotkron(Mati,Matd);
        regularization = lambda*kron(reg,eye(M));
        x = (C'*C + regularization)\(C'*y);
        loss(ite) = norm(C*x-y)^2+x'*regularization*x;
        error(ite) = mean(sign(C*x)~=y);
        W{d} = reshape(x,M,R);
        reg = reg.*(W{d}'*W{d});
        Matd = Matd.*(Mati*W{d});  
    end
end