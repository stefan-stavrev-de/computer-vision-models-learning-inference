% Author: Stefan Stavrev 2013

function sample = sample_gibbs_one (sample_start, Phi, S, D, Values)
    K = length(Values);
    C = length(Phi);
    sample = sample_start;
    for d = 1 : D
        lambda = zeros(K,1);
        for k = 1 : K
            sample(d) = k;
            %Compute unnormalized marginal probability.
            lambda(k) = 1;
            for c = 1 : C
                if(ismember(d, S{c}))
                    args = Values(sample(S{c}));
                    lambda(k) = lambda(k) * Phi{c}(args);
                end
            end
        end
        lambda = lambda ./ sum(lambda);
        sample(d) = randsample(1:K,1,true,lambda);
    end
end