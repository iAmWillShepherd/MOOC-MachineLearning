function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Get Theta 0 and Theta 1 -> n
lengthOfTheta = length(theta);
theta_0 = theta(1);
theta_N = theta(2:length(theta));

% Compute cost J
positiveClass = -y' * log(sigmoid(X * theta));
negativeClass = (1 - y)' * log(1 - sigmoid(X * theta));
J = (1 / m) * sum(positiveClass - negativeClass) + ((lambda / (2 * m)) * (theta_N' * theta_N));

% Compute partial derivative for Theta 0
x_Col_1 = X(:, 1);
grad(1) = (1 / m) * x_Col_1' * (sigmoid(X * theta) - y);

% Compute Theta 1 -> n
x_j_PlusOne = X(: , 2:size(X, 2));
grad(2:length(grad)) = ((1 / m) * x_j_PlusOne' * (sigmoid(X * theta) - y)) + ((lambda / m) * theta(2:length(theta)));

% =============================================================

end
