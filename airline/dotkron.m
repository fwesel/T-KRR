function [CC,Cy] = dotkron(A,B,y)
    [N,DA] = size(A);
    [~,DB] = size(B);
    CC = zeros(DA*DB,DA*DB);
    Cy = zeros(DA*DB,1);
    batchSize = 10000;
    for n = 1:batchSize:N
        idx = min(n+batchSize-1,N);
        temp = repmat(A(n:idx,:),1,DB).*kron(B(n:idx,:), ones(1, DA));
        CC = CC+temp'*temp;
        Cy = Cy+temp'*y(n:idx,:);
    end
end

