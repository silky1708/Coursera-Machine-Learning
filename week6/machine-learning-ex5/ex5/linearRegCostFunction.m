function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
%disp(m);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%disp(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X = [ones(size(X,1)), X];
%disp(size(X));
hypothesis = X*theta;
term1 = hypothesis-y;
term1_sqr = term1.^2;
term1_addn = sum(term1_sqr);
term2_sqr = theta.^2;
term2_addn = sum(term2_sqr)- term2_sqr(1);

J = (term1_addn/(2*m))+((lambda*term2_addn)/(2*m));


grad = (X'*term1)/m;
A = (theta*lambda)/m;
grad(2:end) = grad(2:end)+A(2:end);













% =========================================================================

grad = grad(:);

end
