% Author: Stefan Stavrev 2013

% Description: maximum a posteriori learning of normal distribution.
% Input: x - a row or a column vector, (alpha, beta, gamma, delta) - 
% parameters for the conjugate prior.
% Output: mu - mean, var - variance.
function [mu, var] = map_norm (x, alpha, beta, gamma, delta)
    validate_input(x, alpha, beta, gamma);

    I = length(x);
    mu = (sum(x) + gamma*delta) / (I + gamma);
    var_up = sum((x - mu) .^ 2) + 2*beta + gamma*(delta-mu)^2;
    var_down = I + 3 + 2*alpha;
    var = var_up / var_down;
end

% The input x must be a row or a column vector.
% The parameters alpha, beta, gamma must be strictly greater than zero.
function [] = validate_input (x, alpha, beta, gamma)
    if ~(isrow(x) || iscolumn(x))
        err = 'Invalid input: input must be a row or a column vector.';        
        error(err);
    end
    
    if (alpha <= 0 || beta <= 0 || gamma <= 0)
        a = 'Invalid prior parameters: alpha, beta and gamma must be';
        b = ' strictly greater than zero.';
        err = [a, b];
        error(err);
    end
end