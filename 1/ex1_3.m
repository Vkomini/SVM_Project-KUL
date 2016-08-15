clear all; clc;
%load('/media/sumit/DE20EA8120EA5FCF/MyWorkspace/Dropbox/KUL/MAI/SVM/1/iris.mat/iris.mat');
load('./iris.mat/iris.mat');

mat = [];

for kg=-2:3
    Er = [];
    
    gam = 10^(kg);
    fprintf('gam=%.2f\t',gam);
    for ks=-2:3
        sig2=10^(ks);
        t=1; degree=2;
        %model = {X,Y,'c',gam,[],'lin_kernel','preprocess'};
        %model = {X,Y,'c',gam,[t;degree],'poly_kernel','preprocess'};
        model = {X,Y,'c',gam,sig2,'RBF_kernel','preprocess'};


        %model = {X,Y,'c',[],[],'lin_kernel','csa'};
        %[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm',{10,'misclass'});


        [alpha,b] = trainlssvm(model);

        %hFig = figure(1);
        %set(hFig, 'Position', [10 10 1200 900])
        %plotlssvm(model,{alpha,b});

        estYt = simlssvm(model, {alpha,b},Xt);

        err = sum(estYt~=Yt); 
        %Er = [Er, err];

        mat(kg+3,ks+3) = err/length(Yt)*100;
        
        fprintf(' & %.1f\\%%',err/length(Yt)*100);
        %fprintf('\n$gam = %.3f => on test: \\#misclass = %d/%d, error = %.2f\\%%, MAE: %.3f, MSE: %.3f$\n',gam, err, length(Yt), err/length(Yt)*100,mae(estYt-Yt),mse(estYt-Yt))
        %fprintf('\n gam = %.3f',gam);
        %fprintf('\n on test: #misclass = %d of %d, error rate = %.2f%%\n', err, length(Yt), err/length(Yt)*100)
        %fprintf('MAE: %.3f, MSE: %.3f\n',mae(estYt-Yt),mse(estYt-Yt))
    end
    fprintf('\\\\ \n \\hline \n');
    %hold on;
    %plot(Er);
end

hFig = figure(1);
set(hFig, 'Position', [10 10 1200 900]);
imagesc([-2 3], [-2 3], mat, [0 100]); xlabel('log(sig^2)'); ylabel('log(gamma)');
c = colorbar('Ticks',[0,25,50,75,100], 'TickLabels',{'0%','25%','50%','75%','100%'});
colormap jet;
c.Label.String = 'Error in percentage';

