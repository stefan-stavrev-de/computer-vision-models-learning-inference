% Author: Stefan Stavrev 2013

% The variance is expressed in terms of the standard deviation as
% var = sqrt(var)^2, to constrain var to take only positive values when
% this function is optimized in terms of var.
function f = fit_slr_cost (var, X, w, H)
    I = size(X,2);
    H_inv = diag(1 ./ H);
    covariance = X'*H_inv*X + var*eye(I);
    f = mvnpdf (w, zeros(I,1), covariance);
    f = -log(f);
end