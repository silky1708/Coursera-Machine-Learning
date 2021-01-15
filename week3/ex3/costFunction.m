function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using 
%theta as the
%   parameter for logistic regression and the gradient of the 
%cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to 
%the partial
%               derivatives of the cost w.r.t. each parameter in %theta
%
% Note: grad should have the same dimensions as theta

sum = 0;
hypo =0;
n = size(theta);
theta = theta';
p=0;	q =0;		A=0; B=0; el =0;
for i=1:m,
	
	xi = X(i,:);
	xi = xi';
	p = theta*xi;
	%print(p);
	q = 1+(exp(-1*p));
	hypo = 1/q;
	A  = log(hypo);
	B = log(1-hypo);
	el = y(i);
	A = el*A;
	B = (1-el)*B;
	sum = sum+ A+ B;

for j=1:n,
	grad(j) += (hypo-el)*xi(j);
j++;
end;
	
		
i++;
end;




J = (-1*sum)/m;
grad = grad./m;


% =============================================================

end
