X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(size(X));
%Outliers are added via:
out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));

figure, plot(X,Y,'*')

%% Fixed gam and sig2 (manual)
gam=100;
sig2=0.1;
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
figure, plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'}, {alpha,b});

%% auto tune (normal) mse
globalOptFun = 'csa';
optFun = 'simplex';
[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', globalOptFun}, optFun,'crossvalidatelssvm',{10,'mse'})
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
figure, plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'}, {alpha,b});

%% auto tune (normal) mae
globalOptFun = 'csa';
optFun = 'simplex';
[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', globalOptFun}, optFun,'crossvalidatelssvm',{10,'mae'})
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
figure, plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'}, {alpha,b})

%% Auto tune (robust) Huber mae
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm(model,'simplex',costFun,{10,'mse'},wFun);
model = robustlssvm(model);
figure, plotlssvm(model);


%% Auto tune (robust) hampel mae
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whampel';
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model = robustlssvm(model);
figure, plotlssvm(model);


%% Auto tune (robust) Logistic mae
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'wlogistic';
model = tunelssvm(model,'simplex',costFun,{10,'mse'},wFun);
model = robustlssvm(model);
figure, plotlssvm(model);


%% Auto tune (robust) Myriad mae
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'wmyriad';
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model = robustlssvm(model);
figure, plotlssvm(model);



Fun = {'whampel', 'wlogistic','wmyriad'}