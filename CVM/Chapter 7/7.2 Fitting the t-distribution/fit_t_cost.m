% Author: Stefan Stavrev 2013

% The cost function to be minimized for nu.
function [val] = fit_t_cost (nu, E_hi, E_log_hi)
    nu_half = nu / 2;
    I = size(E_hi,1);
    val = I * (nu_half*log(nu_half) + gammaln(nu_half));
    val = val - (nu_half-1)*sum(E_log_hi);
    val = val + nu_half*sum(E_hi);
    val = -1 * val; % the minus in front of the main sum
end