% Author: Stefan Stavrev 2013

% Description: Dynamic programming (Viterbi algorithm) for chain models.
% Input: U - NxK matrix that contains the unary costs,
%        P - 1xN cell of KxK matrices that contain the pairwise costs.
% Output: w - minimum cost path.
function w = viterbi_chain(U, P)
    N = size(U,1);
    K = size(U,2);
    S = zeros(N,K);
    S(1,:) = U(1,:);
    R = zeros(N,K);
    w = zeros(N,1);
    for n = 2 : N
        for k = 1 : K
            temp = S(n-1,:) + P{n}(k,:);            
            [m, R(n,k)] = min(temp);
            S(n,k) = U(n,k) + m;
        end
    end
    [~, w(N)] = min(S(N,:));
    for n = N:-1:2
        w(n-1) = R(n,w(n));
    end    
end