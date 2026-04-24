% Author: Stefan Stavrev 2013

% Description:  sampling from the Dirichlet distribution.
% Input:        alpha - Dirichlet distribution parameters,
%               N - number of samples to generate.
% Output:       r - random samples from the Dirichlet distribution.
function r = randdir (alpha, N)
    K = length(alpha);
    r = gamrnd(repmat(alpha,N,1),1,N,K);
    r = r ./ repmat(sum(r,2),1,K);
end