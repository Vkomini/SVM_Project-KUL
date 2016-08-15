clear all; clc;
load('./iris.mat/iris.mat');

idx = randperm(size(X,1));

Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

mat = [];

for kg=-2:3
    %Er = [];
    gam = 10^(kg);
    
    for ks=-2:3
        sig2=10^(ks);
        
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
        estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b},Xval);
        
        cv_performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'}, 10,'misclass');
        loo_performance = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass');

        %hFig = figure(1);
        %set(hFig, 'Position', [10 10 800 600])
        %plotlssvm(model,{alpha,b});

        estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b},Xval);
        err = sum(estYval~=Yval); 
        %Er = [Er, err]
        
        fprintf('\ngam = %.2f\tsig2=%.2f',gam,sig2);
        fprintf('\non validation set: #misclass = %d of %d, error rate = %.2f%%\n', err, length(Yval), err/length(Yval)*100)
        fprintf('MAE: %.2f, MSE: %.2f\n',mae(estYval-Yval),mse(estYval-Yval))
        fprintf('Cross-validation error: %.3f \t Leave-one-out error: %.3f\n',cv_performance,loo_performance)
    end
    %hold on;
    %plot(Er);
end

