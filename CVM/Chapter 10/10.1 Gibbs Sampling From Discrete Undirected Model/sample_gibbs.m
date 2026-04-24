% Author: Stefan Stavrev 2013

% Description: Gibbs sampling from discrete undirected model.
% Input: Phi - cell of C functions,
%        S - cell of C arrays which represent the domains for the functions
%            in Phi. For example, S{1} = [1 2] means that the function
%            Phi{1} depends on the variables 1 and 2,
%        D - dimensionality of one sample, also the number of
%            variables in the joint distribution,
%        Values - each of the D variables can take K values,
%        T - number of samples to generate.
% Output: Samples - DxT matrix which contains the samples in its columns.
%                   A sample in this case does not contain directly
%                   the values for the variables. Instead, it contains
%                   indices for values in the vector Values.
function Samples = sample_gibbs (Phi, S, D, Values, T)
    Samples = ones(D,T);    
    for t = 2 : T
        Samples(:,t) = sample_gibbs_one(Samples(:,t-1), Phi, S, D, Values);        
    end
end