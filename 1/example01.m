datasetSize = 2*1000;
split = [0.60, 0.20];
maxPoly = 10;

gam = 50;

%% Generate Dataset
X = [1.5+randn(datasetSize/2,2); -1.5-randn(datasetSize/2,2)];
Y = [ones(datasetSize/2,1); -1*ones(datasetSize/2,1)];

%{
hold on;
plot(X(1:50,1),X(1:50,2),'ro');
plot(X(51:100,1),X(51:100,2),'bo');
hold off;
%}


%% Divide Training, Validation and test set

idx = randperm(size(X,1));
Xtrain = X(idx(1:(split(1)*datasetSize)),:);
Ytrain = Y(idx(1:(split(1)*datasetSize)),:);

%{
hold on;
plot(Xtrain(find(Ytrain>0),1),Xtrain(find(Ytrain>0),2),'ro');
plot(Xtrain(find(Ytrain<0),1),Xtrain(find(Ytrain<0),2),'bo');
hold off;
%}

Xval = X(idx((split(1)*datasetSize)+1:((split(1)+split(2))*datasetSize)),:);
Yval = Y(idx((split(1)*datasetSize)+1:((split(1)+split(2))*datasetSize)),:);

%{
hold on;
plot(Xval(find(Yval>0),1),Xval(find(Yval>0),2),'mo');
plot(Xval(find(Yval<0),1),Xval(find(Yval<0),2),'ko');
hold off;
%}

Xtest = X(idx(((split(1)+split(2))*datasetSize)+1:datasetSize),:);
Ytest = Y(idx(((split(1)+split(2))*datasetSize)+1:datasetSize),:);

%{
hold on;
plot(Xtest(find(Ytest>0),1),Xtest(find(Ytest>0),2),'yo');
plot(Xtest(find(Ytest<0),1),Xtest(find(Ytest<0),2),'go');
hold off;
%}


%% Train LS-SVM

[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,[],'lin_kernel'});

figure, plotlssvm({Xtrain,Ytrain,'c',gam,[],'lin_kernel'},{alpha,b});
hold on;
plot(Xval(find(Yval>0),1),Xtest(find(Yval>0),2),'yo');
plot(Xval(find(Yval<0),1),Xval(find(Yval<0),2),'bo');
xlabel('X label');
ylabel('Y label');
hold off;

estYtrain = simlssvm({Xtrain,Ytrain,'c',gam, [],'lin_kernel'},{alpha,b},Xtrain);
errTrain = sum(Ytrain ~= estYtrain)/(split(1)*datasetSize);
accTrain = 1 - errTrain;

estYval = simlssvm({Xtrain,Ytrain,'c',gam, [],'lin_kernel'},{alpha,b},Xval);
errVal = sum(Yval ~= estYval)/(split(2)*datasetSize);
accVal = 1 - errVal;


for degree=2:maxPoly
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'c', gam, [1;degree], 'poly_kernel'});
    
    %{
    figure, plotlssvm({Xtrain,Ytrain,'c',gam,[1;degree], 'poly_kernel'},{alpha,b});
    hold on;
    plot(Xval(find(Yval>0),1),Xtest(find(Yval>0),2),'y^');
    plot(Xval(find(Yval<0),1),Xval(find(Yval<0),2),'b>');
    xlabel('X label');
    ylabel('Y label');
    hold off;
    %}
    
    estYtrain = simlssvm({Xtrain,Ytrain,'c', gam, [1;degree],'poly_kernel'}, {alpha,b}, Xtrain);
    errTrain = [errTrain; sum(Ytrain ~= estYtrain)/(split(1)*datasetSize)];
    accTrain = [accTrain; 1-errTrain];
    
    estYval = simlssvm({Xtrain,Ytrain,'c',gam, [1;degree],'poly_kernel'},{alpha,b},Xval);
    errVal = [errVal; sum(Yval ~= estYval)/(split(2)*datasetSize)];
    accVal = [accVal; 1-errVal];
end

figure, plot(1:maxPoly,errVal, 'm--');
hold on;
plot(1:maxPoly, errTrain, 'b--')
xlabel('Degrees of polynomial');
ylabel('Fraction of Missclassification');
legend('Validation Error', 'Training Error');
hold off;

figure, plot(1:maxPoly,1 - errVal, 'm--');
hold on;
plot(1:maxPoly, 1 - errTrain, 'b--')
xlabel('Degrees of polynomial');
ylabel('Accuracy');
legend('Validation Accuracy', 'Training Accuracy');
hold off;

%close all;



