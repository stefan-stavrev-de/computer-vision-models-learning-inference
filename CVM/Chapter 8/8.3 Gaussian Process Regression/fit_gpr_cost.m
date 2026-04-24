% Author: Stefan Stavrev 2013

% The variance is expressed in terms of the standard deviation as
% var = sqrt(var)^2, to constrain var to take only positive values when
% this function is optimized in terms of var.
function f = fit_gpr_cost (var, K, w, var_prior)
    I = length(w);
    covariance = var_prior*K + (sqrt(var)^2)*eye(I);
    f = mvnpdf (w, zeros(I,1), covariance);
    f = -log(f);
end