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
%               derivatives of the cost w.r.t. each parameter in %theta

J_init = 0;	grad_init = zeros(length(theta));
[J_, grad_] = costFunction(theta, X, y);
J_init = J_; grad_init = grad_;
sum_of_sqr =0;
sum =0;

for j=2:size(theta),
	
		sum_of_sqr+= theta(j)^2;
		grad(j) = grad_init(j)+((lambda*theta(j))/m);
	
j++;
end;

grad(1) = grad_init(1);
J = J_init +((lambda*sum_of_sqr)/(2*m));


% =============================================================

end
