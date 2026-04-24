% Author: Stefan Stavrev 2013

% Description: Fitting a factor analyzer.
% Input: x - a matrix where each row is one datapoint,
%        K - number of factors.
% Output: mu - 1xD mean vector,
%         phi - DxK matrix containing K factors in its columns,
%         sig - Dx1 vector representing the DxD diagonal matrix Sigma.
function [mu, phi, sig] = fit_fa (X, K, iterations)
    I = size(X,1);
    D = size(X,2);

    % Initialize mu to the data mean.    
    mu = sum(X,1) ./ I;
    
    %Initialize phi to random values.
    rng('default');
    phi = randn(D,K);
    
    % Initialize sig, by setting its diagonal elements to the
    % variances of the D data dimensions.
    x_minus_mu = bsxfun (@minus, X, mu);
    sig = sum (x_minus_mu .^ 2, 1) ./ I;
    
    % The main loop.
    iterations_count = 0;  
    while true
        % Expectation step.
        inv_sig = diag(1 ./ sig);
        phi_transpose_times_sig_inv = phi' * inv_sig;
        temp = inv (phi_transpose_times_sig_inv * phi + eye(K));
        E_hi = temp * phi_transpose_times_sig_inv * x_minus_mu';
        E_hi_hitr = cell (1, I);
        for i = 1 : I
            e = E_hi(:,i);
            E_hi_hitr{i} = temp + e*e';
        end
        
        % Maximization step.
        % Update phi.
        phi_1 = zeros(D,K);
        for i = 1 : I
            phi_1 = phi_1 + (x_minus_mu(i,:)' * E_hi(:,i)');
        end
        phi_2 = zeros(K,K);
        for i = 1 : I
            phi_2 = phi_2 + E_hi_hitr{i};
        end
        phi_2 = inv (phi_2);
        phi = phi_1 * phi_2;
        
        % Update sig.        
        sig_diag = zeros(D,1);
        for i = 1 : I
            xm = x_minus_mu(i,:)';
            sig_1 = xm .* xm;
            sig_2 = (phi * E_hi(:,i)) .* xm;
            sig_diag = sig_diag + sig_1 - sig_2;
        end
        sig = sig_diag ./ I;
        
        iterations_count = iterations_count + 1;        
        disp(['iteration ' num2str(iterations_count)]);        
        if iterations_count == iterations
            break;
        end
    end    
end