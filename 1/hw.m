

X = trainset
Xt = testset

Y = labels_train
Yt = labels_test


%%
model = {X,Y,'c',[],[],'lin_kernel','csa'};
[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'});
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});

plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});
[Ysim,Ylatent] = simlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b},Xt);

roc(Ylatent, Yt)