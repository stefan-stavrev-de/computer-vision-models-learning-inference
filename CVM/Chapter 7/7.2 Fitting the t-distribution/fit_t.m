% Author: Stefan Stavrev 2013

% Description: Fitting the t-distribution.
% Input: x - a matrix where each row is one datapoint,
%        precision - the algorithm stops when the difference between
%                    the previous and the new likelihood is < precision.
%                    Typically this is a small number like 0.01.
% Output: mu  - the mean of the fitted t-distribution,
%         sig - the scale matrix,
%         nu - degrees of freedom.
function [mu, sig, nu] = fit_t (x, precision)
    % Initialize mu to the mean of the dataset.
    I = size (x, 1);
    dataset_mean = sum(x,1) ./ I;
    mu = dataset_mean;
    
    % Initialize sig to the covariance of the dataset.
    D = size (x, 2);
    dataset_variance = zeros (D, D);
    x_minus_dataset_mean = bsxfun (@minus, x, dataset_mean);
    for i = 1 : I
        mat = x_minus_dataset_mean(i,:);
        mat = mat' * mat;
        dataset_variance = dataset_variance + mat;
    end
    dataset_variance = dataset_variance ./ I;
    sig = dataset_variance;
    
    % Initialize degrees of freedom to 1000 (just a random large value).
    nu = 1000;
    
    % The main loop.
    iterations = 0;    
    previous_L = 1000000; % just a random initialization
    delta = zeros (I,1);
    while true
        % Expectation step.
        % Compute delta.
        x_minus_mu = bsxfun (@minus, x, mu);
        temp = x_minus_mu * inv(sig);
        for i = 1 : I
            delta(i) = temp(i,:) * x_minus_mu(i,:)';
        end
        
        % Compute E_hi.
        nu_plus_delta = nu + delta;
        E_hi = (nu + D) ./ nu_plus_delta;
        
        % Compute E_log_hi.
        E_log_hi = psi((nu+D)/2) - log(nu_plus_delta./2);
                
        % Maximization step.
        % Update mu.
        E_hi_sum = sum (E_hi);
        E_hi_times_xi = bsxfun (@times, E_hi, x);
        mu = sum (E_hi_times_xi, 1) ./ E_hi_sum;
        
        % Update sig.
        x_minus_mu = bsxfun (@minus, x, mu);
        sig = zeros (D,D);
        for i = 1 : I
            xmm = x_minus_mu(i,:);
            sig = sig + E_hi(i) * xmm' * xmm;
        end
        sig = sig ./ E_hi_sum;
        
        % Update nu by minimizing a cost function with line search.
		nu = fminbnd(@(nu) fit_t_cost(nu,E_hi, E_log_hi), 0, 1000);
     
        % Compute delta again, because the parameters were updated.
        temp = x_minus_mu * inv(sig);
        for i = 1 : I
            delta(i) = temp(i,:) * x_minus_mu(i,:)';
        end   
        
        % Compute the log likelihood L.
        L = I * (gammaln((nu+D)/2) - (D/2)*log(nu*pi) - ...
            log(det(sig))/2 - gammaln(nu/2));
        s = sum (log(1 + delta./nu)) / 2;
        L = L - (nu+D)*s;
        %disp(L);
        
        iterations = iterations + 1;
        %disp([num2str(previous_L) ' ' num2str(L)]);
        if abs(L - previous_L) < precision
        %if iterations == 50
            %disp(num2str(iterations));
            break;
        end
        
        previous_L = L;
    end    
end