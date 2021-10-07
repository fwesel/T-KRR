%% Spambase Dataset
M = 40;
R = 10;
maxIte = 10;
lambda = 1e-5;
NTrials = 10;
trainErrorCP = zeros(NTrials,1);
testErrorCP = zeros(NTrials,1);
trainErrorKRR = zeros(NTrials,1);
testErrorKRR = zeros(NTrials,1);
trainErrorRFF = zeros(NTrials,1);
testErrorRFF = zeros(NTrials,1);
timeWallKRR = zeros(NTrials,1);
timeWallCP = zeros(NTrials,1);
timeWallRFF = zeros(NTrials,1);
warning('off','all');
for ite = 1:NTrials
    rng(ite);
    X = importdata('spambase.data');
    [~,D] = size(X);
    perm = randperm(size(X,1));
    X = X(perm,:);
    X = X(1:floor(0.9*size(X,1)),:);    %train on 2/3 of the data
    Y = X(:,end);
    X = X(:,1:end-1);
    Y = (Y==1)-(Y==0); 
    XMin = min(X);  XMax = max(X);
    X = (X-XMin)./(XMax-XMin);
    lengthscale = mean(std(X));
    tic;
    WCP = CPLS(X,Y,M,R,lambda,lengthscale,maxIte);
    timeWallCP(ite) = toc;
    trainErrorCP(ite) = mean(Y~=sign(CPPredict(X,WCP,lengthscale)));
    
    % RFF
    tic;
    [ZZ,ZY,W,B] = RFF(X,Y,R*M,lengthscale);
    wRFF = (ZZ+lambda*eye(R*M))\(ZY);
    timeWallRFF(ite) = toc;
    trainErrorRFF(ite) = mean(Y~=sign(RFFPredict(X,W,B)*wRFF));
    
    % KRR
    tic;
    [wKRR,XTrain] = KRR(X,Y,lengthscale,lambda);
    timeWallKRR(ite) = toc;
    trainErrorKRR(ite) = mean(Y~=sign(SE(X,X,lengthscale)*wKRR));
    
    % Test
    X = importdata('spambase.data');
    X = X(perm,:);
    X = X(floor(0.9*size(X,1))+1:end,:);
    Y = X(:,end);
    X = X(:,1:end-1);
    Y = (Y==1)-(Y==0); 
    X = (X-XMin)./(XMax-XMin);
    testErrorCP(ite) = mean(Y~=sign(CPPredict(X,WCP,lengthscale)));
    testErrorRFF(ite) = mean(Y~=sign(RFFPredict(X,W,B)*wRFF));
    testErrorKRR(ite) = mean(Y~=sign(SE(X,XTrain,lengthscale)*wKRR));
end