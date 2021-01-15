function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:(num_movies*num_features)), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta

temp_y = Y.*R;

temp_matrix = zeros(size(R));
for i=1:1:num_users
	soyatheta = Theta(i,:);
	for j=1:1:num_movies
		xi = transpose(X(j,:));
		temp_matrix(j, i) = soyatheta*xi;
	end
end

temp_matrix = temp_matrix.*R;
answer = temp_matrix - temp_y;
answer = answer.^2;
rowvector = sum(answer);
answer = sum(rowvector);
J = answer/2;

sum1 = sum(sum(Theta.^2));
sum2 = sum(sum(X.^2));
summer = (sum1+sum2)*lambda;
J = J + summer/2;

n = size(X, 2);
for i=1:1:num_movies
	column = transpose(R(i,:));
	xi = X(i,:)';
	left = Theta*xi;
	right = Y(i,:)';
	answer = left-right;
	answer = answer.*column;
	for k=1:1:n
		grad_k = answer.*Theta(:,k);
		X_grad(i,k) = sum(grad_k);
	end
	addend = lambda*X(i,:);
	X_grad(i,:) = X_grad(i,:)+addend;

end


for j=1:1:num_users
	column = R(:,j);
	first = X*(Theta(j,:)');
	second = Y(:,j);
	answer = first-second;
	answer = answer.*column;

	for k=1:1:n
		grad_k = answer.*X(:,k);
		Theta_grad(j, k) = sum(grad_k);
	end
	addend = lambda*Theta(j,:);
	Theta_grad(j,:) = Theta_grad(j,:)+addend;
end















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
