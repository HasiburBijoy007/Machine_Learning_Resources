function [J grad] = Mynn(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1), X];
         
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%FORWARD PROPAGATION.........
Z2 = X * Theta1';
A2 = sigmoid(Z2);
A2 = [ones(m, 1), A2];

Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

y_new = zeros(m, num_labels);   %(5000, 10)
for t=1:m
  y_new(t,y(t)) = 1; 
end

%COST FUNCTION WITHOUT REGULARIZATION...
loss = ((-y_new) .* log(A3)) - ((1-y_new) .* log(1-A3));
J = (1/m) * sum(sum(loss));


%REGULARIZATION COMPUTATION.....
T1 = Theta1(:, 2:size(Theta1, 2));
T2 = Theta2(:, 2:size(Theta2, 2));
reg = (lambda/(2*m)) * ( sum(sum(power(T1, 2)))  +  sum(sum(power(T2, 2))) );


%COST FUNCTION WITH REGULARIZATION...
J = J + reg;


%FORWARD PROPAGATION & BACK-PROPAGATION........
for i=1:m
  a1 = X(i,:);
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(1,1), a2];

  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  
  z2 = [ones(1,1), z2];

  D3 = a3 - y_new(i, :);
  D2 = (D3 * Theta2) .* sigmoidGradient(z2);
  D2 = D2(2:end);
  
  Theta2_grad = Theta2_grad + (D3' * a2);
  Theta1_grad = Theta1_grad + (D2' * a1);
   
end 



%Obtain the (unregularized) gradient for the neural network cost function.....
%Theta1_grad = (1/m) * Theta1_grad;
%Theta2_grad = (1/m) * Theta2_grad;


%Obtain the (regularized) gradient for the neural network cost function....
reg_term1 = (lambda/m) * Theta1(:, 2:end);
reg_term2 = (lambda/m) * Theta2(:, 2:end);

Theta1_grad(:, 1) = (1/m) * Theta1_grad(:, 1);
Theta1_grad(:, 2:end) = ((1/m) * Theta1_grad(:, 2:end)) + reg_term1;
	
Theta2_grad(:, 1) = (1/m) * Theta2_grad(:, 1);
Theta2_grad(:, 2:end) = ((1/m) * Theta2_grad(:, 2:end))+ reg_term2;


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
