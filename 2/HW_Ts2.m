clear
load('/media/sumit/DE20EA8120EA5FCF/MyWorkspace/KUL/MAI/SVM/2/santafe.mat')

%zmean = mean(Z);
%Z = (Z-zmean)/std(Z);

test_size = 100;
Ztrain = Z(1:length(Z)-test_size);
Ztest = Z(length(Z)-test_size+1:end);

zmean = mean(Ztrain);
Ztrain = (Ztrain-zmean)/std(Ztrain);

zmean = mean(Ztest);
Ztest = (Ztest-zmean)/std(Ztest);


Xt = Ztest;

order = 50;
Xu = windowize(Ztrain,1:(order+1));
Ytra = Xu(:,end);
Xtra = Xu(:,1:order);

%globalOptFun = 'csa';
optFun = 'simplex';
[gam,sig2,cost] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'}, optFun,'crossvalidatelssvm',{10,'mae'});
%gam = 10;sig2=10;
[alpha,b] = trainlssvm({Xtra,Ytra,'f',gam,sig2, 'RBF_kernel'});

Xs = Ztrain(end-order+1:end,1);

prediction = predict({Xtra, Ytra, 'f', gam, sig2, 'RBF_kernel'}, Xs, 100);
figure, plot([prediction Xt]);
legend('Predict', 'Original');
cost
%cost_crossval = crossvalidate({Xtra,Ytra,'f',gam,sig2},10, 'mse')
%cost_crossval = crossvalidate({prediction,Xt,'f',gam,sig2},10, 'mae')
mse(prediction-Xt)
mae(prediction-Xt)

