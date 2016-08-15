gam = 10;
sig2 = 10;

idx = randperm(size(X,1));
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80),:);
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

op=[];
for n=1:100
    sig2=n;
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
    estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);

    pos=0;
    for i=1:20
        if(estYval(i)==Yval(i))
            pos=pos+1;
        end
    end

    op = [op,(pos/20)*100.0];
end

plot(1:100,op)

cross10 = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'},10,'misclass');
Leaveoneout = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass');
