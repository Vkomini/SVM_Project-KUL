clear all;
clc;

%% Ripley Data set

load('D:\MyWorkspace\KUL\MAI\SVM\1\ripley.mat\ripley.mat');
gam = 10;
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});
%}




%% Breast

load('D:\MyWorkspace\KUL\MAI\SVM\1\breast.mat\breast.mat');
[gam, sig2, cost] = tunelssvm({trainset,labels_train,'c',[],[],'lin_kernel','csa'}, 'simplex', 'crossvalidatelssvm', {10,'misclass'});
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'});
%figure, plotlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b});
%figure, plotlssvm({testset,labels_test,'c',gam,[],'lin_kernel'},{alpha,b});

%[Ysim,Ylatent] = simlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b},trainset);
%roc(Ylatent, labels_train);
[Ysim,Ylatent] = simlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b},testset);
roc(Ylatent, labels_test);


[gam, sig2, cost] = tunelssvm({trainset,labels_train,'c',[],[],'RBF_kernel','csa'}, 'simplex', 'crossvalidatelssvm', {10,'misclass'});
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'});
%figure, plotlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b});
%figure, plotlssvm({testset,labels_test,'c',gam,sig2,'RBF_kernel'},{alpha,b});


%[Ysim,Ylatent] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b},trainset);
%roc(Ylatent, labels_train);
[Ysim,Ylatent] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b},testset);
roc(Ylatent, labels_test);
%}




%% Diabetes
%{
load('D:\MyWorkspace\KUL\MAI\SVM\1\diabetes.mat\diabetes.mat');
[gam, sig2, cost] = tunelssvm({trainset,labels_train,'c',[],[],'lin_kernel','csa'}, 'simplex', 'crossvalidatelssvm', {10,'misclass'});
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'});
%figure, plotlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b});
%figure, plotlssvm({testset,labels_test,'c',gam,[],'lin_kernel'},{alpha,b});

%[Ysim,Ylatent] = simlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b},trainset);
%roc(Ylatent, labels_train);
[Ysim,Ylatent] = simlssvm({trainset,labels_train,'c',gam,[],'lin_kernel'},{alpha,b},testset);
roc(Ylatent, labels_test);


[gam, sig2, cost] = tunelssvm({trainset,labels_train,'c',[],[],'RBF_kernel','csa'}, 'simplex', 'crossvalidatelssvm', {10,'misclass'});
[alpha,b] = trainlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'});
%figure, plotlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b});
%figure, plotlssvm({testset,labels_test,'c',gam,sig2,'RBF_kernel'},{alpha,b});


%[Ysim,Ylatent] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b},trainset);
%roc(Ylatent, labels_train);
[Ysim,Ylatent] = simlssvm({trainset,labels_train,'c',gam,sig2,'RBF_kernel'},{alpha,b},testset);
roc(Ylatent, labels_test);
%}


