function [] = tuneHyperParamsBayes(Xtrain, Ytrain)

gam = 10; sig2 = 0.025;
%{
[~, alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~, gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~, sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

[model, alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[model, gam] = bay_optimize(model,2);
[model, sig2] = bay_optimize(model,3);

[model, alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[model, gam] = bay_optimize(model,2);
[model, sig2] = bay_optimize(model,3);

[model, alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[model, gam] = bay_optimize(model,2);
[model, sig2] = bay_optimize(model,3);
%}

[model, alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[model, gam] = bay_optimize(model,2);
[model, sig2] = bay_optimize(model,3);

sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure')

cost_crossval = crossvalidate({Xtrain,Ytrain,'f',gam,sig2},10)
cost_loo = leaveoneout({Xtrain,Ytrain,'f',gam,sig2})
gam
sig2
figure, plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});

