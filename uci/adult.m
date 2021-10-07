%% Adult Dataset
M = 40;
R = 10;
maxIte = 10;
lambda = 1e-5;
NTrials = 10;
trainErrorCP = zeros(NTrials,1);
testErrorCP = zeros(NTrials,1);
trainErrorRFF = zeros(NTrials,1);
testErrorRFF = zeros(NTrials,1);
timeCP = zeros(NTrials,1);
timeRFF = zeros(NTrials,1);
warning('off','all');
for ite = 1:NTrials
    rng(ite);
    X = readmatrix('adult.csv');
    perm = randperm(size(X,1));
    X = X(perm,:);
    X = X(1:floor(0.9*size(X,1)),:);
    Y = X(:,end);
    X = X(:,1:end-1);

    Y = (Y==1)-(Y==0); 
    XMin = min(X);  XMax = max(X);
    X = (X-XMin)./(XMax-XMin);
    lengthscale = mean(std(X));
    tic;
    WCP = CPLS(X,Y,M,R,lambda,lengthscale,maxIte);
    timeCP(ite) = toc;
    trainErrorCP(ite) = mean(Y~=sign(CPPredict(X,WCP,lengthscale)));
    
    
    % RFF
    tic;
    [ZZ,ZY,W,B] = RFF(X,Y,M*R,lengthscale);
    wRFF = (ZZ+lambda*eye(M*R))\(ZY);
    timeRFF(ite) = toc;
    trainErrorRFF(ite) = mean(Y~=sign(RFFPredict(X,W,B)*wRFF));
    
    % Test
    X = readmatrix('adult.csv');
    X = X(perm,:);
    X = X(floor(0.9*size(X,1))+1:end,:);
    Y = X(:,end);
    X = X(:,1:end-1);
    Y = (Y==1)-(Y==0); 
    X = (X-XMin)./(XMax-XMin);
    testErrorCP(ite) = mean(Y~=sign(CPPredict(X,WCP,lengthscale)));
    testErrorRFF(ite) = mean(Y~=sign(RFFPredict(X,W,B)*wRFF));
end