function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


%COST FUNCTION WITHOUT REGULARIZATION......
pred = X * Theta';
error = pred - Y;
sqerror = (pred - Y) .^ 2;
J = (1/2) * sum(sum(R .* sqerror)); 

%GRADIENTS WITHOUT REGULARIZATION..........
X_grad = (R .* error) * Theta;
Theta_grad = (R .* error)' * X;



%COST FUNCTION WITH REGULARIZATION......
reg_term1 = (lambda/2) * (X .^ 2);
reg_term2 = (lambda/2) * (Theta .^ 2);
J = J + sum(sum(reg_term1)) + sum(sum(reg_term2));

%GRADIENTS WITHOUT REGULARIZATION..........
reg_Xgrad = lambda * X;
reg_Thetagrad = lambda * Theta;
X_grad = X_grad + reg_Xgrad;
Theta_grad = Theta_grad + reg_Thetagrad;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MOVIE RATINGS BY ME................
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
movieList = loadMovieList();

function my_ratings = ratings_me(movieList)

%  Initialize my ratings
my_ratings = zeros(1682, 1);
my_ratings(1) = 4;
my_ratings(98) = 2;
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s  \n', my_ratings(i), movieList{i});
    end
end

end 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%TRAIN MY RS MODEL......
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('ex8_movies.mat');

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

%  After training the model, you can now make recommendations by computing
%  the predictions matrix.

p = X * Theta';
my_predictions = p(:,1) + Ymean;
movieList = loadMovieList();
[r, ix] = sort(my_predictions, 'descend');

for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s', my_predictions(j), movieList{j});
end


for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s', my_ratings(i), movieList{i});
    end
end
