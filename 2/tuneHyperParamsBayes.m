function [] = tuneHyperParamsBayes(Xtrain, Ytrain)

gamO = 0.1
sig2O = 1
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

[model, alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gamO,sig2O},1);
[model, gam] = bay_optimize(model,2);
[model, sig2] = bay_optimize(model,3);

sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');

cost_crossval = crossvalidate({Xtrain,Ytrain,'f',gam,sig2},10)
cost_loo = leaveoneout({Xtrain,Ytrain,'f',gam,sig2})
gamO
sig2O
gam
sig2
figure, plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});
%figure, plot(sig2e)

