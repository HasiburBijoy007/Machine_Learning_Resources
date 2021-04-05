function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% Initialize some useful values

% The first column is the population of a city and 
% the second column is the profitt of a food truck in that city.

m = length(y); % number of training examples %97!


%AS We have only one feature in X (as X is (97,1)),  
%Then we must have only one theta........

%AS We Will add X0(97,1) with Our Originial X(97,1) and X will be (97,2)! 
%(See GradientDescentFunction)
%Then we must have Two thetas...........

%BUT in both Cases, theta or thetas initialize from ZERO!

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

pred = X * theta;
loss = (pred - y) .^ 2;
J = (1/(2*m)) .* sum(loss);


% =========================================================================

end
