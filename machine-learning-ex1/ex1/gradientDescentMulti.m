function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    err = X * theta - y;
  %  t0 = theta(1);
  %  t1 = theta(2);
  %  t2 = theta(3);
  %  t0 = t0 - alpha * (1/m) * sum(err .* X(:, 1));
  %  t1 = t1 - alpha * (1/m) * sum(err .* X(:, 2));
  %  t2 = t2 - alpha * (1/m) * sum(err .* X(:, 3));
  %  theta(1) = t0;
  %  theta(2) = t1;
  %  theta(3) = t2;
    theta = theta - alpha * (1 / m) * (X' * err);
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
