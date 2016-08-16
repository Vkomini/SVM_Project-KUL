clear
%load('/media/sumit/DE20EA8120EA5FCF/MyWorkspace/KUL/MAI/SVM/2/santafe.mat')
load('santafe.mat')



zmean = mean(Z);
zsigma = std(Z);
Ztrain = (Z-zmean)/zsigma;

order = 110;
Xu = windowize(Ztrain,1:(order+1));
Ytra = Xu(:,end);
Xtra = Xu(:,1:order);

%globalOptFun = 'csa';
optFun = 'simplex';
[gam,sig2,cost] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'}, optFun,'crossvalidatelssvm',{10,'mse'});
%gam = 10;sig2=10;
[alpha,b] = trainlssvm({Xtra,Ytra,'f',gam,sig2, 'RBF_kernel'});

Zt = (Ztest(1:order) - zmean)/zsigma;
Xs = Zt(1:order,1);

prediction = predict({Xtra, Ytra, 'f', gam, sig2, 'RBF_kernel'}, Xs, max(size(Ztest))-order);

prediction = (prediction*zsigma) + zmean;
figure, plot([prediction Ztest((order+1):end)]);
title(sprintf('Order: %d, MSE: %.2f',order, mse(prediction-Ztest((order+1):end))))
cost


legend('Predict', 'Original');

mse(prediction-Ztest((order+1):end))
mae(prediction-Ztest((order+1):end))

