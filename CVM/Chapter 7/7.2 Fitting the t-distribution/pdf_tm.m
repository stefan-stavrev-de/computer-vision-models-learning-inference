% Author: Stefan Stavrev 2013

% Multivariate t-distribution pdf.
function [px] = pdf_tm (x, mu, sig, nu)
    D = length(mu);
    % gammaln is used instead of gamma to avoid overflow.
    % gamma((nu+D)/2)/gamma(nu/2) == exp(gammaln((nu+D)/2)-gammaln(nu/2)).
    c = exp(gammaln((nu+D)/2) - gammaln(nu/2));
    c = c / ((nu*pi)^(D/2) * sqrt(det(sig)));
        
    I = size(x,1);
    delta = zeros (I,1);
    x_minus_mu = bsxfun (@minus, x, mu);
    temp = x_minus_mu * inv(sig);
    for i = 1 : I
        delta(i) = temp(i,:) * x_minus_mu(i,:)';
    end
    
    px = 1 + (delta ./ nu);
    px = px .^ (-nu-D)/2;
    px = px * c;
end