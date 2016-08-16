function params = tuneHyperParams(X,Y)

params = [];

for i=1:2
    for j=1:2
        if(i==1)
            optFun = 'gridsearch';
        else
            optFun = 'simplex';
        end
        
        if(j==1)
            globalOptFun = 'csa';
        else
            globalOptFun = 'ds';
        end
        
        tic;
        [gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', globalOptFun}, optFun,'crossvalidatelssvm',{10,'mse'});
        time = toc;
        cost_crossval = crossvalidate({X,Y,'f',gam,sig2},10);
        cost_loo = leaveoneout({X,Y,'f',gam,sig2});
        %str = sprintf('Gamma = %lf , Sigma = %lf, Cost = %f, Cross_Val = %f, LeaveOneOut = %f, Time = %f',gam,sig2,cost,cost_crossval, cost_loo, time);
        params = [params; [gam,sig2,cost,cost_crossval, cost_loo, time]];
        %disp(optFun);
        %disp(globalOptFun);
        %disp(str);
        [alpha,b] = trainlssvm({X,Y,'f',gam,sig2});
        %figure, plotlssvm({X,Y,'f',gam,sig2},{alpha,b});
    end
end

t=1;
for i=1:2
    for j=1:2
        if(i==1)
            optFun = 'gridsearch';
        else
            optFun = 'simplex';
        end
        
        if(j==1)
            globalOptFun = 'csa';
        else
            globalOptFun = 'ds';
        end
        
        disp(globalOptFun);
        disp(optFun);
        str = sprintf('Gamma = %.3f , Sigma = %.3f, Cost = %f, Cross_Val = %f, LeaveOneOut = %f, Time = %f',params(t,1),params(t,2),params(t,3),params(t,4), params(t,5), params(t,6));
        disp(str);
        t=t+1;
    end
end

end