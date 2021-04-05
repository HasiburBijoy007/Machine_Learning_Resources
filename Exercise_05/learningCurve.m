function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);


% ---------------------- Sample Solution ----------------------
%HERE,
%theta = Optimal theta gained by fmincg Optimization Algorithm
%             from trainLinearReg(X_train, y_train, lambda)!!! 
%AND
%We don't need to return grad from linearRegCostFunction!
%  because 'grad' won't need here!!!


for i=1:m
  X_train = X(1:i, :);
  y_train = y(1:i);
  theta = trainLinearReg(X_train, y_train, lambda);
  error_train(i) = linearRegCostFunction(X_train, y_train, theta, 0);
  error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end


% -------------------------------------------------------------

% =========================================================================

end
