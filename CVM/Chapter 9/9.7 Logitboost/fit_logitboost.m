% Author: Stefan Stavrev 2013

% Description: Logitboost.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        Alpha - (D+1)xM matrix which contains the parameters for the
%                the weak classifiers in its columns.
% Output: predictions - 1xI_test row vector which contains the predicted
%                       class values for the input data in X_test.
function predictions = fit_logitboost (X, w, X_test, Alpha)
    K = 10;
    I = size(X,2);
    a = zeros(I,1);
    phi_0 = 0;
    phi = zeros(K,1);   
    c = zeros(K,1);
    M = size(Alpha,2);
    
    for k = 1 : K
        % Find best weak classifier.
        current_max = -1;
        for m = 1 : M
            value = 0;
            for i = 1 : I
                f = Alpha(:,m)' * X(:,i);
                if(f < 0)
                    f = 0;
                else
                    f = 1;
                end
                value = value + (a(i)-w(i)) * f;
            end
            value = value * value;
            if(value > current_max)
                current_max = value;
                c(k) = m;
            end
        end
        
        % Remove effect of offset parameters.
        a = a - phi_0;
        
        % Optimize.
        options = optimset('GradObj','on');
        initial_x = [0; 0];
        x = fminunc(@(x) fit_logitboost_cost(x, X, w, a, Alpha(:,c(k))), initial_x, options);
        phi_0 = x(1);
        phi(k) = x(2);
        
        % Compute new activation.
        for i = 1 : I
            f = Alpha(:,c(k))' * X(:,i);
            if(f < 0)
                f = 0;
            else
                f = 1;
            end
            a(i) = a(i) + phi_0 + phi(k) * f;
        end 
    end
    
    I_test = size(X_test,2);
    predictions = zeros(1,I_test);
    for i = 1 : I_test
        act = phi_0;
        for k = 1 : K
            f = Alpha(:,c(k))' * X_test(:,i);
            if (f < 0)
                f = 0;
            else
                f = 1;
            end
            act = act + phi(k) * f;
        end
        predictions(i) = sigmoid(act);
    end
end