%% Banana Dataset Plots
M = 12;
RSet = [2,4,6,M^2];
lambda = 1e-6;
lengthscale = 0.5;
NIte = 10;
%% Generate Plots
close all
rng('default');
X = readmatrix('banana.csv');
N = size(X,1);
X = X(randperm(size(X,1)),:);
Y = (X(:,end)==1)-(X(:,end)==2);
X = X(:,1:2);
XMin = min(X);  XMax = max(X);
X = (X-XMin)./(XMax-XMin);

NPlot = 1000;
X1Plot = linspace(0,1,NPlot);
[X1Plot,X2Plot] = meshgrid(X1Plot,X1Plot);
XPlot = [X1Plot(:),X2Plot(:)];

% Low-rank exact
Z = ones(size(X,1),1);
ZPlot = ones(size(XPlot,1),1);
for d = 1:size(X,2)
    Z = dotkron(Z,features(X(:,d),M,lengthscale));
    ZPlot = dotkron(ZPlot,features(XPlot(:,d),M,lengthscale));
end
wHilbert = (Z'*Z+lambda*eye(M^2))\(Z'*Y);
scorePlotHilbert = sign(ZPlot*wHilbert);

plotIdx = 0;
for R = RSet
    rng(plotIdx);
    plotIdx = plotIdx+1;
    W = CPLS(X,Y,M,R,lambda,lengthscale,NIte);
    scorePlotCP = sign(CPPredict(XPlot,W,lengthscale));
    
    figure(plotIdx);
    fig = gcf;
    hold on
    s1 = scatter(X(Y==1,1),X(Y==1,2),36,[238, 28, 37]/255,'filled');
    s2 = scatter(X(Y==-1,1),X(Y==-1,2),36,[1, 90, 162]/255,'filled');
    s1.MarkerFaceAlpha = 0.25;
    s2.MarkerFaceAlpha = 0.25;
    c1 = contour(X1Plot,X2Plot,reshape(scorePlotCP,size(X1Plot)),[0 0],'black','LineWidth',1.5);
    c2 = contour(X1Plot,X2Plot,reshape(scorePlotHilbert,size(X1Plot)),[0 0],'black','LineWidth',1.5,'LineStyle','--');
    xticks(-1:0.2:1);
    yticks(-1:0.2:1);
    axis equal
    axis off
    hold off
    filename = 'banana'+string(M^2)+'frequencies'+string(R)+'rank'+'.pdf';
%     exportgraphics(fig,filename,'BackgroundColor','none','ContentType','vector');
end