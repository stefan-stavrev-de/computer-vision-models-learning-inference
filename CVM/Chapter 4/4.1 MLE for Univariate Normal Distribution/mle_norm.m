% Author: Stefan Stavrev 2013

% Description: maximum likelihood learning of normal distribution.
% Input: x - a row or a column vector.
% Output: mu - mean, var - variance.
function [mu, var] = mle_norm (x)
    validate_input(x);
    
    I = length(x);
    mu = sum(x) / I;
    var = sum((x - mu) .^ 2) / I;
end

% The input must be a row or a column vector.
function [] = validate_input (x)
    if ~(isrow(x) || iscolumn(x))
        err = 'Invalid input: input must be a row or a column vector.';        
        error(err);
    end
end