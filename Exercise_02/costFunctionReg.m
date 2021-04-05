function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 


% Initialize some useful values

m = length(y);
J = 0;
grad = zeros(size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


%  fdata = load('ex2data2.txt');
%  tX = fdata(:, [1, 2]);
%  fy = fdata(:, 3);
%  m = length(y);
%  fX = [ones(m,1), tX];
%  theta = zeros(3,1);
%  costFunctionReg(theta, fX, fy, lambda=0.1);


%COST FUNCTION...........

z = X * theta;
pred = sigmoid(z);
reg = lambda/(2*m) * (sum(power(theta, 2)) - power(theta(1), 2));
loss = ((-y) .* log(pred)) - ((1-y) .* log(1-pred));
J = ((1/m) * sum(loss)) + reg;


%GRADIENT COMPUTATION..........

grad(1) = (1/m) .* sum((pred - y) .* X(:, 1));
for i = 2:numel(theta)
  grad(i) = ((1/m) .* sum((pred - y) .* X(:, i))) + ((lambda/m) * theta(i));
end

% =============================================================

end
