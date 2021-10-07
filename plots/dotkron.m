function y = dotkron(L,R)
    [r1,c1] = size(L);   [r2,c2] = size(R);
    if r1 ~= r2
        error('Matrices should have equal rows!');
    else
        if r1 > 1e5
            y = zeros(r1,c1*c2);
            batchSize = 100;
            for n = 1:batchSize:r1
                idx = min(n+batchSize-1,r1);
                y(n:idx,:) = repmat(L(n:idx,:),1,c2).*kron(R(n:idx,:), ones(1, c1));
            end
        else
            y = repmat(L,1,c2).*kron(R, ones(1, c1));
        end
    end
end