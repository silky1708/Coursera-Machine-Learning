function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% array of values for which test needs to be done are as follows- 
% 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
array = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
temp_C = 0.01;
temp_sigma = 0.01;
min_error = 15;

for i=1:8
	C = array(i);
	for j=1:8
		sigma = array(j);
		mode = svmTrain(X, y, C, @gaussianKernel, sigma, 1e-3, 5);
		prediction = svmPredict(mode, Xval, sigma);
		err = mean(double(prediction ~= yval));
		if(err< min_error)
			temp_C = C;
			temp_sigma = sigma;
			min_error = err;
		end
	end
end

C = temp_C;
sigma = temp_sigma;



% =========================================================================

end
