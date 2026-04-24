% Author: Stefan Stavrev 2013

% Gamma distribution pdf.
function [px] = pdf_gamma (x, alpha, beta)
    px = beta^alpha / gamma(alpha);
    px = px * exp(x .* (-beta)) .* x.^(alpha-1);
end