% Author: Stefan Stavrev 2013

% Linear kernel function.
% Input: x_i - a column vector,
%        x_j - a column vector.
function f = kernel_linear (x_i, x_j)
    f = x_i' * x_j;
end