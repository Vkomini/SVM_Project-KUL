clear all; clc;

load('./iris.mat/iris.mat');

idx = randperm(size(X,1));

Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

model = {X,Y,'c',[],[],'RBF_kernel','ds'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

cv_performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'}, 10,'misclass');
loo_performance = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass');

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
[estYval, latentY] = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b},Xval);
err = sum(estYval~=Yval)/length(Yval);

roc(latentY,Yval)