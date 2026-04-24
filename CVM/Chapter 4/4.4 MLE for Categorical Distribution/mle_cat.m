% Author: Stefan Stavrev 2013

% Description: maximum likelihood learning of categorical distribution.
% Input: x - a row or a column vector,
%        K - number of categories.
% Output: theta - K categorical distribution parameters.
function [theta] = mle_cat (x, K)
    validate_input (x, K);

    [counts] = hist(x, 1:K);
    theta = counts ./ sum(counts);
end

% The input x must be a row or a column vector, K must be scalar.
function [] = validate_input (x, K)
    if ~(isrow(x) || iscolumn(x))
        err = 'Invalid input: x must be a row or a column vector.';        
        error(err);
    end
    
    if ~(isscalar(K))
        err = 'Invalid input: K must be a scalar value.';
        error(err);
    end
end