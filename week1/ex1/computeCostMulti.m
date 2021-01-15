function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

cost = zeros(m,1);
i =1;
for i = 1:m,
	xi = X(i,:);
	xi = xi';
	hypo = theta' * xi;
	cost(i) = (hypo-y(i))^2;
i++;
end;

sum =0 ;
j = 1;
for j = 1:m,
sum += cost(j);
j++;
end;

J= sum/(2*m);




% =========================================================================

end
