gam = 5; sig2 = 100;
X = 10.*rand(100,3)-3;
Y = cos(X(:,1)) + cos(2*(X(:,1))) + 0.3.*randn(100,1);
figure, plot(X,Y,'*')
[selected, ranking, costs] = bay_lssvmARD({X,Y,'class',gam,sig2});
ranking
selected
costs
%figure, plot([selected])
%[dimensions, ordered, costs, sig2s] =  bay_lssvmARD({X,Y,'class',gam,sig2});