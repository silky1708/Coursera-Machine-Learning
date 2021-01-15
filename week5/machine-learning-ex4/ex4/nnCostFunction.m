function [J grad] = nnCostFunction(nn_params, input_layer_size,hidden_layer_size, num_labels,X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Y = ones(m,1);
X = [Y, X];
q = 0; Del1 = zeros(size(Theta1)); Del2 = zeros(size(Theta2));
y_matrix = zeros(m, num_labels);
hypothesis = zeros(m, num_labels);

Del1 = zeros(size(Theta1));
Del2 = zeros(size(Theta2));

for j=1:m
	label = y(j);
	y_matrix(j,label) = 1;

	x = X(j,:);
	hidden_layer_act = sigmoid( Theta1*x');
	hidden_layer_act = [1; hidden_layer_act];
	a_3 = sigmoid(Theta2*hidden_layer_act);

	del_3 = a_3-y_matrix(j,:)';
	del_2 = (transpose(Theta2(:,2:end))*del_3).*sigmoidGradient( Theta1*x');
	Del1 = Del1 + del_2*x;
	Del2 = Del2+ del_3*transpose(hidden_layer_act);

	hypothesis(j,:) = a_3';
end

b = ones(m, num_labels)-y_matrix;
a = log(hypothesis);
c = log(ones(m, num_labels)-hypothesis);
q = trace(a'*y_matrix)+trace(b'*c);
q = (-1*q)/m;
	sum1 = sum(Theta1.*Theta1);
	sum2 = sum(Theta2.*Theta2);
	sum1(1)=0;	sum2(1)=0;
	J = q +((sum(sum1)+sum(sum2))*lambda)/(2*m);


%for i=1:m
%	j = y(i);
%	vec = ones(size(a_3));
%	vec = vec-a_3;
%	t = log(vec);	q = sum(t);
%	q =q-t(j);
%	q = q +log(a_3(j));

%	del_3 = a_3;
%	del_3(j) = del_3(j)-1;
%	

%	
%	

%end
one = [zeros(size(Theta1,1),1), Theta1(:,2:end) ];
two = [zeros(size(Theta2,1),1), Theta2(:,2:end) ];

Theta1_grad = (Del1/m)+((lambda*one)/m);
Theta2_grad = (Del2/m)+((lambda*two)/m);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
