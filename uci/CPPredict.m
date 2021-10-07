function score = CPPredict(X, W, hyperparameters)
    [N,D] = size(X);
    M = size(W{1},1);
    score = ones(N,1);
    for d = 1:D
        score = score.*(features(X(:,d),M,hyperparameters)*W{d});
    end
    score = sum(score,2);
end