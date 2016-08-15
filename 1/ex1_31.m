clear all; clc;

load('./iris.mat/iris.mat');

gam = 0.1;
sig2 = 20;

idx = randperm(size(X,1));

Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

erMat = [];
cvMat = [];

for kg=-1:3
    gam = 10^(kg);
    fprintf('gam=%.2f\t',gam);
    for ks=-1:2
        
    sig2=10^(ks);
    
    [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
    estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b},Xval);
    
    %performance = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'}, 10,'misclass');
    performance = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass');

    err = sum(estYval~=Yval);
    erMat(kg+2,ks+2) = err/length(Yval)*100;
    cvMat(kg+2,ks+2) = performance;
    
    %fprintf(' & %.1f\\%%',err/length(Yval)*100);
    fprintf(' & %.3f ',performance);
    end
    fprintf('\\\\ \n \\hline \n');
end

%{
hFig = figure(1);
set(hFig, 'Position', [10 10 1200 900]);
imagesc([-1 2], [-1 3], erMat,[0 50]); xlabel('log(sig^2)'); ylabel('log(gamma)');
c = colorbar('Ticks',[0,10,20,30,40,50], 'TickLabels',{'0%','10%','20%','30%','40%','50%'});
colormap jet;
c.Label.String = 'Error in percentage';
%}

hFig = figure(2);
set(hFig, 'Position', [10 10 1200 900]);
imagesc([-1 2], [-1 3], cvMat, [0 0.5]); xlabel('log(sig^2)'); ylabel('log(gamma)');
c = colorbar('Ticks',[0,0.10,0.20,0.30,0.40,0.50], 'TickLabels',{'0%','10%','20%','30%','40%','50%'});
colormap jet;
c.Label.String = 'Error in percentage';



