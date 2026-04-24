% Author: Stefan Stavrev 2013

% Description: Gaussian process regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        var_prior - scale factor for the prior spherical covariance,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        kernel - the kernel function.
% Output: mu_test - a vector of size (I_test x 1), such that mu_test(i)
%                   is the mean of the distribution P(w|x*) for the test
%                   data example x* = X_test(:,i),
%         var_test - a vector of size (I_test x 1), such that var_test(i)
%                    is the variance of the distribution P(w|x*) for the 
%                    test data example x* = X_test(:,i).
function [mu_test, var_test] = fit_gpr (X, w, var_prior, X_test, kernel)
    I = size(X,2);
    I_test = size(X_test,2);

    % Compute K[X,X].    
    K = zeros(I,I);
    for i=1:I
        for j=1:I
            K(i,j) = kernel(X(:,i), X(:,j));
        end
    end
    
    % Compute K[X_test,X].
    K_test = zeros(I_test, I);
    for i=1:I_test
        for j=1:I
            K_test(i,j) = kernel(X_test(:,i), X(:,j));
        end
    end

    % Compute the variance. Use the range [0,variance of world values].
    % Constrain var to be positive, by expressing it as var=sqrt(var)^2,
    % that is, the standard deviation squared.
    mu_world = sum(w) / I;
    var_world = sum((w - mu_world) .^ 2) / I;
    var = fminbnd (@(var) fit_gpr_cost (var, K, w, var_prior), 0, var_world);
    
    % Compute A_inv.
    A_inv = inv (K + (var/var_prior)*eye(I));
    
    % Compute the mean for each test example.
    mu_temp_1 = K_test * w;
    K_test_A_inv = K_test * A_inv;
    mu_temp_2 = K_test_A_inv * K * w;
    c = var_prior/var;
    mu_test = c * (mu_temp_1 - mu_temp_2);
    
    % Compute the variance for each test example.    
    var_test = repmat(var,I_test,1);
    for i = 1 : I_test
        x_star = X_test(:,i);
        part1 = kernel(x_star, x_star);
        part2 = K_test_A_inv(i,:) * K_test(i,:)';
        var_test(i) = var_test(i) + var_prior * (part1 - part2);
    end
end