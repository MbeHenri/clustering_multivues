close all;
load('TC_DS.mat');

%>> loading data
Y = gt;
X = { X1' , X2'};

%>> hyperparameters

% begin constant for regularising learning of cluster indicator for each view
beta = 1;
% begin constant for regularising learning of global cluster indicator
lambda = 1;

%>> execution of model on data
k = size(unique(gt),1);
islocal_1=1; % (boolean 0 or 1)
islocal_2=1; % (boolean 0 or 1)
result = MVGL(X, k, beta, lambda, islocal_1, islocal_2);

%>> compute metrics with label
[ACC, NMI, PUR] = ClusteringMeasure(gt,result{4});