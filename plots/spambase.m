%% Spambase Dataset convergence plot
M = 40;
R = 10;
lambda = 1e-5;
NIte = 10;
warning('off','all');

rng('default');
X = readmatrix('spambase.csv');
perm = randperm(size(X,1));
X = X(perm,:);
X = X(1:floor(0.9*size(X,1)),:); 
Y = X(:,end);
X = X(:,1:end-1);
Y = (Y==1)-(Y==0);

XMin = min(X);  XMax = max(X);
X = (X-XMin)./(XMax-XMin);

lengthscale = mean(std(X));
[~,loss,misclassificationRate] = CPLS(X,Y,M,R,lambda,lengthscale,NIte);
%% Plot
lossPlot = loss./loss(1);   % normalized loss
idx = 1:numel(misclassificationRate);
dx = 2*(size(X,2)-1);
hold on
plot(idx,lossPlot,'black','LineWidth',1.5);
xlim([0,idx(end)]);
xticks(0:2*dx:idx(end))
xticklabels({'0','2','4','6','8','10'})
yticks(0.5:0.1:1)
ylabel('Loss','interpreter','latex','FontSize',20);
xlabel('Number of sweeps','interpreter','latex','FontSize',20)
grid on 
axis square
hold off
% exportgraphics(gcf,'spambaseLoss.pdf','BackgroundColor','none','ContentType','vector');

close all
hold on
plot(idx,misclassificationRate,'black','LineWidth',1.5);
xlim([0,idx(end)]);
xticks(0:2*dx:idx(end))
xticklabels({'0','2','4','6','8','10'})
yticks(0.0:0.1:0.5);
ylabel('Misclassification rate','interpreter','latex','FontSize',20);
xlabel('Number of sweeps','interpreter','latex','FontSize',20)
grid on
axis square
hold off
% exportgraphics(gcf,'spambaseMisclassificationRate.pdf','BackgroundColor','none','ContentType','vector');
