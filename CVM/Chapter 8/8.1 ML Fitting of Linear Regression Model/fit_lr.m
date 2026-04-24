% Author: Stefan Stavrev 2013

% Description: ML Fitting of linear regression model.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example.
% Output: phi - a (D+1)x1 vector containing the linear function coefficients,
%         sig - variance.
function [phi, sig] = fit_lr (X, w)
    % Compute phi.
    phi = inv(X*X')*X*w;
    
    % Compute sig.
    I = size(X,2);
    temp = w - X'*phi;
    sig = (temp' * temp) / I;
end