%its for  costfunctionreg  and  plotDecisionBoundary.........

fdata = load('ex2data2.txt');
tX = fdata(:, [1, 2]);
fy = fdata(:, 3);
m = length(y);
fX = [ones(m,1), tX];
theta = zeros(3,1);
costFunctionReg(theta, fX, fy, lambda=0.1);