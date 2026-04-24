% Author: Stefan Stavrev 2013

% Description: Sparse linear regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I).
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        nu - degrees of freedom typically nu<0.001,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions.
% Output: mu_test - a vector of size (I_test x 1), such that mu_test(i)
%                   is the mean of the distribution P(w|x*) for the test
%                   data example x* = X_test(:,i),
%         var_test - a vector of size (I_test x 1), such that var_test(i)
%                    is the variance of the distribution P(w|x*) for the 
%                    test data example x* = X_test(:,i).
function [mu_test, var_test] = fit_slr (X, w, nu, X_test)
    D = size(X,1) - 1;
    I = size(X,2);
    I_test = size(X_test,2);
    
    % Initialize H.
    H = ones(D+1,1);
    H_old = zeros(D+1,1);
    
    % Precompute. No need to compute these each iteration.
    X_w = X*w;
    X_t = X';
    X_Xt = X*X_t;
    
    % The main loop.
    iterations_count = 0;
    precision = 0.0001;
    while true
        % Compute the variance. Use the range [0,variance of world values].
        % Constrain var to be positive, by expressing it as
        % var=sqrt(var)^2, that is, the standard deviation squared.
        mu_world = sum(w) / I;
        var_world = sum((w - mu_world) .^ 2) / I;
        var = fminbnd (@(var) fit_slr_cost (var, X, w, H), 0, var_world);
        
        % Update sig and mu.
        sig = inv(X_Xt/var + diag(H));
        mu = (sig * X_w) / var;
        
        % Update H.
        H = H .* diag(sig);
        H = nu + 1 - H;
        H = H ./ (mu.^2 + nu);
        H(1) = 1; % make sure the first dimension stays constant.
                
        iterations_count = iterations_count + 1;        
        disp(['iteration ' num2str(iterations_count)]);
        disp(H);
        stop = all(abs(H-H_old) < precision);
        if stop == true
            break;
        end
        
        % Save H for the next iteration.
        H_old = H;
    end
    
    % Prune step. Remove row d in X and element d in H, if H(d)>1.
    selector = H < 1;
    X = X(selector,:);
    X_test = X_test(selector,:);
    H = H(selector);
    
    % Compute A_inv.
    H_inv = diag(1 ./ H);
    H_inv_X = H_inv * X;    
    temp = inv(X' * H_inv_X + var*eye(I));
    A_inv = H_inv - H_inv_X * temp * X' * H_inv;
        
    % Compute the mean for each test example.
    temp2 = X_test' * A_inv;
    mu_test = (temp2 * X * w) ./ var;
    
    % Compute the variance for each test example.    
    var_test = repmat(var,I_test,1);
    for i = 1 : I_test
        var_test(i) = var_test(i) + temp2(i,:) * X_test(:,i);
    end    
end