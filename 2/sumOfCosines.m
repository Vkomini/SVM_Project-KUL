clc;

%X = (-10:0.1:10)';
%Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);

Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

%plotSigGam([1 10 100 1000], [0.01 0.1 1], Xtrain, Ytrain, Xtrain, Ytrain);

%tuneHyperParams(Xtrain, Ytrain);

tuneHyperParamsBayes(Xtrain, Ytrain);