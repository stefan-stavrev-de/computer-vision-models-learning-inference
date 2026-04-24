% Author: Stefan Stavrev 2013

% Description: MAP learning of categorical distribution.
% Input: x - a row or a column vector,
%        alpha - Dirichlet prior.
% Output: theta - K categorical distribution parameters.
function [theta] = map_cat (x, alpha)
    validate_input (x);

    I = length(x);
    K = length(alpha);
    [counts] = hist (x, 1:K);
    theta = (counts - 1 + alpha) ./ (I - K + sum(alpha));
end

% The input x must be a row or a column vector.
function [] = validate_input (x)
    if ~(isrow(x) || iscolumn(x))
        err = 'Invalid input: x must be a row or a column vector.';        
        error(err);
    end
end