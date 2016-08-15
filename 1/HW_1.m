clear all; clc;

%load('./ripley.mat/ripley.mat');
%load('./breast.mat/breast.mat');
load('./diabetes.mat/diabetes.mat')

Xt=trainset;
Yt=labels_train;
X=testset;
Y=labels_test;

idx = randperm(size(Xt,1));

Xtrain = Xt(idx(1:0.8*length(Xt)),:);
Ytrain = Yt(idx(1:0.8*length(Xt)));
Xval = Xt(idx(0.8*length(Xt)+1:length(Xt)),:);
Yval = Yt(idx(0.8*length(Xt)+1:length(Xt)));

%% Linear Model

[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'c',[],[],'lin_kernel','preprocess', 'csa'}, 'simplex','crossvalidatelssvm',{10,'misclass'});

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'lin_kernel', 'preprocess'});
%plotlssvm({Xtrain,Ytrain,'c',gam,sig2,'lin_kernel'},{alpha,b});

[estYval L_lin_val]= simlssvm({Xtrain,Ytrain,'c',gam,sig2,'lin_kernel','preprocess'}, {alpha,b},Xval);
lin_val_err = (sum(estYval~=Yval)/length(Yval))*100;

[estYval L_lin_test]= simlssvm({Xtrain,Ytrain,'c',gam,sig2,'lin_kernel','preprocess'}, {alpha,b},X);
lin_test_err = (sum(estYval~=Y)/length(Y))*100;


%% RBF Kernel

[gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'c',[],[],'RBF_kernel', 'csa'}, 'simplex','crossvalidatelssvm',{10,'misclass'});

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
%plotlssvm({Xtrain,Ytrain,'c',gam,sig2,'lin_kernel'},{alpha,b});

[estYval L_rbf_val]= simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b},Xval);
rbf_val_err = (sum(estYval~=Yval)/length(Yval))*100;

[estYval L_rbf_test]= simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b},X);
rbf_test_err = (sum(estYval~=Y)/length(Y))*100;


fprintf('\nPercentage Error on validation set(Linear Kernel): %.3f%%\n',lin_val_err);
fprintf('Percentage Error on test set(Linear Kernel): %.3f%%\n',lin_test_err);

fprintf('\nPercentage Error on validation set(RBF Kernel): %.3f%%\n',rbf_val_err);
fprintf('Percentage Error on test set(RBF Kernel): %.3f%%\n',rbf_test_err);

roc(L_lin_val, Yval)
roc(L_lin_test, Y)

roc(L_rbf_val, Yval)
roc(L_rbf_test, Y)
