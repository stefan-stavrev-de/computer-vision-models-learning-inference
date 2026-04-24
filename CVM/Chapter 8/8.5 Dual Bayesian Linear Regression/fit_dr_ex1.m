% Author: Stefan Stavrev 2013

% Generate data ----------------------------------------------------------
I = 70; % training examples
D = 100; % dimensionality
I_test = 10; % test examples

% Create I training data points in D-dimensional space. Column == data point.
X_train = randn(D,I);
X_train = [ones(1,I); X_train];

% Evaluate a plane equation in D-dimensional space on the training data points.
phi = ones(D+1,1); % plane equation coefficients
w = X_train' * phi; % evaluate the plane function

% Prepare the test data.
X_test = randn(D,I_test);
X_test = [ones(1,I_test); X_test]; 
% ------------------------------------------------------------------------


% Dual linear regression -------------------------------------------------
kernel = @kernel_linear;
var_prior = 6;
[mu_test, var_test] = fit_dr (X_train, w, var_prior, X_test, kernel);

% Compare the original and the learned models.
original_model_predictions = X_test' * phi;
learned_model_predictions = mu_test;
disp([original_model_predictions learned_model_predictions]);
% ------------------------------------------------------------------------