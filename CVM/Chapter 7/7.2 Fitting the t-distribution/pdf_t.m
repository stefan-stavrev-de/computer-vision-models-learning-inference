% Author: Stefan Stavrev 2013

% Univariate t-distribution pdf.
function [px] = pdf_t (x, mu, sig, nu)
    px = gamma((nu+1)/2) / (sqrt(nu*pi*sig) * gamma(nu/2));
    px = px * (1 + (x-mu).^2 ./ (nu*sig)) .^ ((-nu-1)/2);
end