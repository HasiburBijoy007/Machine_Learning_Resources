function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1), X];            %(5000, 401)

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);       %(5000,1)


Z1 = X * Theta1';
A1 = sigmoid(Z1);
A1 = [ones(m, 1), A1];

Z2 = A1 * Theta2';
A2 = sigmoid(Z2);

[dummy, p] = max(A2, [], 2);
% =========================================================================


end
