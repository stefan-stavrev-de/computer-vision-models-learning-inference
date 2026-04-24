% Author: Stefan Stavrev 2013

% Description: Cost function for incremental logistic regression.
% Input: x - (2+D+1)x1 column vector subject to minimization,
%        X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        a - a Ix1 vector which contains the activations for all the
%            training examples.
% Output: f - the value of the function,
%         g - the gradient.
function [f, g] = fit_inclr_cost (x, X, w, a)
    I = size(X,2);
    D = size(X,1) - 1;
    f = 0;
    g = zeros(2+D+1, 1);
    temp1 = x(3:end)' * X;
    for i = 1 : I
        y = sigmoid(a(i) + x(1) + x(2) * atan(temp1(i)));
        if w(i) == 1
            f = f - log(y);
        else
            f = f - log(1-y);
        end
        
        temp2 = y - w(i);        
        g(1) = g(1) + temp2;
        g(2) = g(2) + temp2 * atan(temp1(i));
        g(3:end) = g(3:end) + temp2 * x(2) * (1 / (1 + temp1(i)^2)) * X(:,i);
    end
end