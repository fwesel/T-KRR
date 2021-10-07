%% Airline Dataset
M = 40;
NTrials = 10;
maxIte = 10;
NSet = [10000, 100000, 1000000, 5929413];
RSet = [5,10,15,20];
train = zeros(NTrials,numel(NSet),numel(RSet));
test = zeros(NTrials,numel(NSet),numel(RSet));
wallTime = zeros(NTrials,numel(NSet),numel(RSet));
warning('off','all');
NIdx = 0;
for N = NSet
    NIdx = NIdx+1;
    RIdx = 0;
    lambda = 100/N;
    for R = RSet
        RIdx = RIdx+1;
        for ite = 1:NTrials
            rng(ite+R+N);
            X = readmatrix('airline.csv');  % data already preprocessed
            perm = randperm(size(X,1));
            X = X(perm,:);
            X = X(1:N,:);
            X = X(1:floor(2*N/3),:);    %train on 2/3 of the data
            Y = X(:,end);
            X = X(:,1:end-1);
          
            YMean = mean(Y);    YStd = std(Y);
            XMin = min(X);  XMax = max(X);
            Y = (Y-YMean)./YStd;
            X = (X-XMin)./(XMax-XMin);
            lengthscale = mean(std(X));
            
            % Train
            disp("N: "+string(N)+" R: "+string(R)+" ite: "+string(ite));
            tic;tic;
            WCP = CPLS(X,Y,M,R,lambda,lengthscale,maxIte);
            wallTime(ite,NIdx,RIdx) = toc;toc;
            train(ite,NIdx,RIdx) = mean((Y-CPPredict(X,WCP,lengthscale)).^2);

            % Test
            clear X Y
            X = readmatrix('airline.csv');
            X = X(perm,:);
            X = X(1:N,:);
            X = X(floor(2*N/3)+1:end,:);    %test on 1/3 of the data
            Y = X(:,end);
            X = X(:,1:end-1);
            X = (X-XMin)./(XMax-XMin);
            Y = (Y-YMean)./YStd;
            test(ite,NIdx,RIdx) = mean((Y-CPPredict(X,WCP,lengthscale)).^2);
            disp('Test error: '+string(test(ite,NIdx,RIdx)));
%             save('airline.mat','train','test','wallTime');
        end
    end
end