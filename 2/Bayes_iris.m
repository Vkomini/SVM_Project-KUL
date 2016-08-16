clear;
load iris;

gamma = [0.5 5 50];
sigma2 = [0.075 0.75 7.5];

mg = max(size(gamma))
ms = max(size(sigma2))

for i=1:mg
    for j=1:ms
        gam = gamma(i); sig2 = sigma2(j)
        bay_modoutClass({X,Y,'c',gam,sig2},'figure');
    end
end