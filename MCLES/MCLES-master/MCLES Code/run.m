close all;

%>> loading data
load('MSRC-v1.mat');
gt = Y;
for i=1:4
    X{i} = X{i}';
end

%>> hyperparameters
% maximum of iterations (unsigned integer)
maxIters = 30;

% dimension of unified representation (unsigned integer)
d = 70;

% constant for regularising learning unified representation (unsigned double)
alpha = 0.8;

% constant for regularising learning of global similarity matrix (unsigned double)
beta = 0.4;

% constant for regularising learning of cluster indicator matrix (unsigned double)
gamma = 0.004;

%>> execution of the model on data
k = size(unique(gt),1);
result = MCLES(X, k, alpha, beta, d, gamma, maxIters);

%>> compute metrics with using label
[ACC, NMI, PUR] = ClusteringMeasure(gt,result{5});