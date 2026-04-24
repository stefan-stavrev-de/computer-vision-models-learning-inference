% Author: Stefan Stavrev 2013

% Description: Bayesian approach to univariate normal distribution.
% Input: x - a row or a column vector, prior parameters, and test data.
% Output: posterior parameters, and prediction values for the test data.
function [alpha_post, beta_post, gamma_post, delta_post, x_prediction]...
    = by_norm(x, alpha_prior, beta_prior, gamma_prior, delta_prior, x_test)
    validate_input(x, alpha_prior, beta_prior, gamma_prior, x_test);

    % Compute posterior parameters.
    I = length(x);
    alpha_post = alpha_prior + I/2;
    beta_post = sum(x.^2)/2 + beta_prior + (gamma_prior*delta_prior^2)/2 ...
         - (gamma_prior*delta_prior + sum(x))^2 / (2*(gamma_prior + I));
    gamma_post = gamma_prior + I;
    delta_post = (gamma_prior*delta_prior + sum(x)) / (gamma_prior + I);

    % Compute intermediate parameters.
    alpha_int = alpha_post + 0.5;
    beta_int = (x_test.^2)/2 + beta_post + (gamma_post*delta_post^2)/2 - ...
        (gamma_post*delta_post + x_test).^2 / (2*gamma_post + 2);
    gamma_int = gamma_post + 1;

    % Predict values for x_test.
    temp1 = sqrt(gamma_post) * (beta_post^alpha_post) * gamma(alpha_int);
    x_prediction_up = repmat(temp1, 1, length(x_test));
    x_prediction_down = sqrt(2*pi) * sqrt(gamma_int) * gamma(alpha_post)...
        * beta_int.^alpha_int;
    x_prediction =  x_prediction_up ./ x_prediction_down;
end

% The inputs x and x_test must be row or column vectors.
% The parameters alpha, beta, gamma must be strictly greater than zero.
function [] = validate_input (x, alpha, beta, gamma, x_test)
    if ~(isrow(x) || iscolumn(x))
        err = 'Invalid input: input must be a row or a column vector.';        
        error(err);
    end
    
    if ~(isrow(x_test) || iscolumn(x_test))
        err = 'Invalid test data: must be a row or a column vector.';        
        error(err);
    end
    
    if (alpha <= 0 || beta <= 0 || gamma <= 0)
        a = 'Invalid prior parameters: alpha, beta and gamma must be';
        b = ' strictly greater than zero.';
        err = [a, b];
        error(err);
    end
end