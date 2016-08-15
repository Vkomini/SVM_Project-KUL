function [t] = plotSigGam(gamma, sigma, Xtrain, Ytrain, Xtest, Ytest)

disp(legend);

gn = max(size(gamma))
sn = max(size(sigma))

trainPlot = [];
Alpha = [];

t = 1;
for i=1:gn
    for j=1:sn
        
        gam = gamma(i);
        sig2 = sigma(j);
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});
        
        trainPlot = [trainPlot; [gam, sig2, b]];
        Alpha = [Alpha; alpha ];
        
        %subplot(gn,sn,t), plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b});
        
        %subplot(gn,sn,t), plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b});
        %subplot(gn,sn,t), plot(H,'go');        
        YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);

        subplot(gn,sn,t), plot(Xtest,Ytest,'b.');        
        subplot(gn,sn,t), hold on;
        subplot(gn,sn,t), plot(Xtest,YtestEst,'r+');
        subplot(gn,sn,t), legend('Ytrain', 'YtrainEst');
        str = sprintf('Gamma = %.3f , and Sigma = %.3f',gam,sig2);
        subplot(gn,sn,t), title(str, 'FontSize', 10);
        t = t+1;
    end
end

%{
for i=1:gn
    for j=1:sn
        
        gam = gamma(i);
        sig2 = sigma(j);
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

        figure, plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b});
        hold on;
        YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);
        plot(Xtest,Ytest,'b.');
        hold on;
        plot(Xtest,YtestEst,'r+');
        legend('Ytest','YtestEst');
        str = sprintf('Gamma = %.3f , and Sigma = %.3f',gam,sig2);
        title(str, 'FontSize', 10);
    end
end
%}

end