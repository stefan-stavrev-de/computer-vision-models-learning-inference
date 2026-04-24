% Author: Stefan Stavrev 2013

% Description: Cost function for logitboost.
% Input: x - (2+D+1)x1 column vector subject to minimization,
%        X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        a - a Ix1 vector which contains the activations for all the
%            training examples,
%        alpha - (D+1)x1 vector which contains the parameters for the
%                current best weak classifier.
% Output: f - the value of the function,
%         g - the gradient.
function [f, g] = fit_logitboost_cost (x, X, w, a, alpha)
    I = size(X,2);
    f = 0;
    g = zeros(2,1);
    for i = 1 : I
        temp = alpha' * X(:,i);
        if(temp < 0)
            temp = 0;
        else
            temp = 1;
        end
        y = sigmoid(a(i) + x(1) + x(2) * temp);
        if w(i) == 1
            f = f - log(y);
        else
            f = f - log(1-y);
        end
        
        temp2 = y - w(i);        
        g(1) = g(1) + temp2;
        g(2) = g(2) + temp2 * temp;
    end
end