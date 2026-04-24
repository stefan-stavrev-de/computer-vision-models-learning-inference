% Author: Stefan Stavrev 2013

% Description: Bayesian approach to categorical distribution.
% Input: x - a row or a column vector,
%        alpha_prior - prior parameters.
% Output: alpha_post - posterior parameters,
%         prediction - predictions for all categories.
function [alpha_post, prediction] = by_cat (x, alpha_prior)
    validate_input (x); 
    
    % Compute posterior.
    K = length (alpha_prior);
    [counts] = hist (x, 1:K);
    alpha_post = alpha_prior + counts;

    % Predict.
    prediction = alpha_post ./ sum(alpha_post);
end

% The input x must be a row or a column vector.
function [] = validate_input (x)
    if ~(isrow(x) || iscolumn(x))
        err = 'Invalid input: x must be a row or a column vector.';        
        error(err);
    end
end