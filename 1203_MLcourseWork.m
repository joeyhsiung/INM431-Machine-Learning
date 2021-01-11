%% Load data
clear;clc;close all
frauds = readtable('D:\OneDrive - City, University of London\cityU\1-Machine Learning\CourseWork\data\CreditCardFraudDetection/creditcard.csv');
colNames = categorical(frauds(:,1:30).Properties.VariableNames);
colNames = reordercats(colNames);
%% Data processing & Primary statistics
% • check missing value and describtive statistics
summary(frauds)
missingvalues = sum(ismissing(frauds));

summaryTable = varfun(@(x) [mean(x);std(x);skewness(x);min(x);max(x)],frauds);
summaryTable.Properties.RowNames = {'mean' 'std' 'skewness' 'min' 'max'};
summaryTable.Properties.VariableNames = extractAfter(summaryTable.Properties.VariableNames,'Fun_');
%% • View hisgrams
varnames = strrep(frauds.Properties.VariableNames,'_','');
figure(1)
for i = 1:30
    subplot(5,6,i)
    histogram(frauds{:,i},30)
    title(varnames(i));
end
%% • View correlation matrix
%%%%% if this part doesnt work pealse run second part

%%%%% First part
% view correlation matrix
% Produce the input matrix data 
varnames = strrep(frauds.Properties.VariableNames,'_','');
fraud = table2array(frauds);
C = corrcoef(fraud);
% Set [min,max] value of C to scale colors
clrLim = [-1,1];  % range
% Set the  [min,max] of diameter where 1 consumes entire grid square
diamLim = [0.1, 1];

% Compute center of each circle 
% This assumes the x and y values were not entered in imagesc()
x = 1 : 1 : size(C,2); % x edges
y = 1 : 1 : size(C,1); % y edges
[xAll, yAll] = meshgrid(x,y); 
% Set color of each rectangle
% Set color scale
cmap = hsv(256); 
Cscaled = (C - clrLim(1))/range(clrLim); % always [0:1]
colIdx = discretize(Cscaled,linspace(0,1,size(cmap,1)));
% Set size of each circle
% Scale the size in the same way we scale the color
diamSize = Cscaled * range(diamLim) + diamLim(1); 
% Create figure
fh = figure(); 
ax = axes(fh); 
hold(ax,'on')
colormap(ax,'hsv');
% Create circles
theta = linspace(0,2*pi,50); % the smaller, the less memory req'd.
h = arrayfun(@(i)fill(diamSize(i)/2 * cos(theta) + xAll(i), ...
    diamSize(i)/2 * sin(theta) + yAll(i), cmap(colIdx(i),:)),1:numel(xAll)); 
axis(ax,'equal')
axis(ax,'tight')
set(gca, 'XTick', 1:31); % x-axis bins
set(gca, 'YTick', 1:31); % y-axis bins
xticklabels(varnames);
xtickangle(305);
yticklabels(varnames);
set(ax,'YDir','Reverse')
colorbar()
caxis(clrLim);
% The heatmap clearly shows which all variable are multicollinear in nature,
% and which variable have high collinearity with the target variable.

%%%%% Second part
figure(5)
set(gcf,'color','w');
fraud = table2array(frauds);
corr_matrix = corrcoef(fraud);
imagesc(corr_matrix); 
set(gca, 'XTick', 1:31); % x-axis bins
set(gca, 'YTick', 1:31); % y-axis bins
xticklabels(varnames);
xtickangle(305);
yticklabels(varnames);
set(gca, 'YTickLabel');
title('Feature and Class Correlation Matrix');
colormap('hsv'); 
colorbar;
hold off

%% • Data partition

dataMatrix = frauds{:,1:30}; % convert traing data to matrix
TagMatrix = table2array(frauds(:,end)); % convert traing data labes to matrix

[m,n] = size(frauds) ;
P = 0.7; %dividing original data into training and testing dataset(0.7:0.3)
idx = randperm(m);
Training = frauds(idx(1:round(P*m)),:); 
Testing = frauds(idx(round(P*m)+1:end),:);

trainDataMatrix = dataMatrix(idx(1:round(P*m)),:);
trainTagMatrix = TagMatrix(idx(1:round(P*m)),:);

testDataMatrix = dataMatrix(idx(round(P*m)+1:end),:);
testTagMatrix = TagMatrix(idx(round(P*m)+1:end),:);
%% Model 1: Random Forest
% • Bayesian hyperparameter optimization
t1=datetime('now');
rng('default'); % For reproducibility
minLS = optimizableVariable('minLS',[1,20],'Type','integer');
numPTS = optimizableVariable('numPTS',[1,30],'Type','integer');
hyperparametersRF = [minLS; numPTS];

results = bayesopt(@(params)oobErrRF(params,trainDataMatrix,trainTagMatrix),hyperparametersRF,...
    'AcquisitionFunctionName','expected-improvement-plus','Verbose',1);
t2=datetime('now');
totalTime=t2-t1;

%% • Training RF Model
bestHyperparameters = results.XAtMinObjective;
Mdl = TreeBagger(100,trainDataMatrix,trainTagMatrix,'OOBPredictorImportance','on','Method','classification',...
    'MinLeafSize',bestHyperparameters.minLS,...
    'NumPredictorstoSample',bestHyperparameters.numPTS);

%% • Plot feature importance
figure(3)
color = [0.13 0.54 0.13]; %color
imp = Mdl.OOBPermutedPredictorDeltaError;
b = bar(colNames,imp,'FaceColor',color);
b.FaceColor = 'flat';
ylabel('RF Predictor importance estimates');
xlabel('Predictors');
grid on

%% • Classification error as a funtion of number of trees
figure(4)
oobErrorBaggedEnsemble = oobError(Mdl);
plot(oobErrorBaggedEnsemble);
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';

%% • Confusion matrix
RFpredictedY_train = predict(Mdl, trainDataMatrix);
RFpredictedY_train = str2double(RFpredictedY_train);
figure(6)
cm = confusionchart(trainTagMatrix,RFpredictedY_train);
cm.Normalization = 'row-normalized';
%% Model 2 Logistic Regression
% • Lasso for generalized linear model
%Construct a regularized binomial regression using 25 Lambda values and 10-fold cross validation.
rng('default') % for reproducibility
[B,FitInfo] = lassoglm(trainDataMatrix,trainTagMatrix,'binomial',...
    'NumLambda',25,'CV',10);
lassoPlot(B,FitInfo,'PlotType','CV');    
legend('show','Location','best')
% The plot identifies the minimum-deviance point with a green circle and dashed line as a function of the
% regularization parameter Lambda. The blue circled point has minimum deviance plus no more than one standard deviation.
%%
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log');
% shhow nonzero model coefficients as a function of the regularization parameter Lambda.
%% 
indx = FitInfo.Index1SE;
B0 = B(:,indx);
nonzeros = sum(B0 ~= 0);
% Find the number of nonzero model coefficients at the Lambda value with minimum deviance plus one standard deviation poin
%%
cnst = FitInfo.Intercept(indx);
B1 = [cnst;B0];
%%
figure(8)
preds = glmval(B1,trainDataMatrix,'logit');
histogram(trainTagMatrix - preds) 
title('Residuals from lassoglm model')
% plot residuals from lassoglm model
%%
predictors = find(B0); 
% indices of nonzero predictors
%% • Train model
LRmd = fitglm(trainDataMatrix,trainTagMatrix,'linear',...
    'Distribution','binomial','PredictorVars',predictors);
%%
figure(9)
plotResiduals(LRmd)
% plot residuals
%%
LRpredictedY_train_temp = predict(LRmd,trainDataMatrix);
LRpredictedY_train = (LRpredictedY_train_temp>=0.5);
LRpredictedY_train = double(LRpredictedY_train);
%% • Confusion matrix
figure(10)
cm2 = confusionchart(trainTagMatrix,LRpredictedY_train);
cm2.Normalization = 'row-normalized';

%% • Feature importance
figure(7)
color2 = [0.58 0.84 0.93]; %color
b = bar(colNames,abs(B0),'FaceColor',color2);
b.FaceColor = 'flat';
ylabel('Predictor importance estimates');
xlabel('Predictors');
grid on

%% Testing and Comparesion
% RF Confusion matrix
RFpredictedY_test = predict(Mdl,testDataMatrix);
RFpredictedY_test = str2double(RFpredictedY_test);
figure(11)
cm3 = confusionchart(testTagMatrix,RFpredictedY_test);
cm3.Normalization = 'row-normalized';

RFcon = confusionmat(testTagMatrix,RFpredictedY_test);
rfAcc = (RFcon(1,1)+RFcon(2,2))/sum(sum(RFcon));
rfRec = RFcon(2,2)/(RFcon(2,1)+RFcon(2,2));   
rfPre = RFcon(2,2)/(RFcon(1,2)+RFcon(2,2));      
randf  = 2/(1/rfPre + 1/logrec);
fprintf('Random Forest Accuracy = %d',randf);
%%
% LR Confusion matrix
LRpredictedY_test_temp = predict(LRmd,testDataMatrix);
LRpredictedY_test = (LRpredictedY_test_temp>=0.5);
LRpredictedY_test = double(LRpredictedY_test);
figure(12)
cm4 = confusionchart(testTagMatrix,LRpredictedY_test);
cm4.Normalization = 'row-normalized';

LRcon = confusionmat(testTagMatrix,LRpredictedY_test);
logAcc = (LRcon(1,1)+LRcon(2,2))/sum(sum(LRcon));
logRec = LRcon(2,2)/(LRcon(2,1)+LRcon(2,2));   
logPre = LRcon(2,2)/(LRcon(1,2)+LRcon(2,2));      
logit  = 2/(1/logpre + 1/rfPre);
fprintf('logistic Regression Accuracy = %d',logit);
%%
% • ROC Curves and AUC
curves = zeros(4,1); labels = cell(4,1);
[RFtrainX,RFtrainY,RFtrainT,RFtrainAUC] = perfcurve(trainTagMatrix,RFpredictedY_train,1);
[LRtrainX,LRtrainY,LRtrainT,LRtrainAUC] = perfcurve(trainTagMatrix,LRpredictedY_train_temp,1);
[RFtestX,RFtestY,RFtestT,RFtestAUC] = perfcurve(testTagMatrix,RFpredictedY_test,1);
[LRtestX,LRtestY,LRtestT,LRtestAUC] = perfcurve(testTagMatrix,LRpredictedY_test_temp,1);

figure(13)
plot(RFtrainX,RFtrainY,'LineWidth',2,'DisplayName',strcat('TrianRF-AUC=',num2str(RFtrainAUC)))
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC and AUC for Training and Testing Set')

hold on
plot(LRtrainX,LRtrainY,'LineWidth',2,'DisplayName',strcat('TrianLR-AUC=',num2str(LRtrainAUC)))
plot(RFtestX,RFtestY,'LineWidth',2,'DisplayName',strcat('TestRF-AUC=',num2str(RFtestAUC)))
plot(LRtestX,LRtestY,'LineWidth',2,'DisplayName',strcat('TestLR-AUC=',num2str(LRtestAUC)))
hold off
lgd = legend('Location','southeast');
lgd.NumColumns = 2;

%% Summary
modelNames = {'LogisticRegression','RandomForest'};
perfMeasures = {'Accuracy', 'Recall', 'Precision', 'f1 score', 'AUC'};
res = [logAcc,rfAcc;logRec,rfRec;logPre,rfPre;logit,randf;...
    LRtestAUC(1)*100,RFtestAUC(1)*100];
restable= array2table(res,'RowNames',perfMeasures,'VariableNames',modelNames);
disp(restable);
%% Function
function oobErr = oobErrRF(params,trainDataMatrix,trainTagMatrix)
%oobErrRF Trains random forest and estimates out-of-bag ensemble error
%   oobErr trains a random forest of 100 regression trees using the
%   predictor data in X and the parameter specification in params, and then
%   returns the out-of-bag quantile error based on the median. X is a table
%   and params is an array of OptimizableVariable objects corresponding to
%   the minimum leaf size and number of predictors to sample at each node.
randomForest = TreeBagger(100,trainDataMatrix,trainTagMatrix,'Method','classification',...
    'OOBPrediction','on','MinLeafSize',params.minLS,...
    'NumPredictorstoSample',params.numPTS);
oobErr = mean(oobError(randomForest));
end