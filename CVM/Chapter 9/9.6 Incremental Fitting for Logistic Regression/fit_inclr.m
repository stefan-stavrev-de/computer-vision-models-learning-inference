% Author: Stefan Stavrev 2013

% Description: Incremental fitting for logistic regression.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions.
% Output: predictions - 1xI_test row vector which contains the predicted
%                       class values for the input data in X_test.
function predictions = fit_inclr (X, w, X_test)
    K = 10;
    phi_0 = 0;
    phi = zeros(K,1);
    I = size(X,2);
    a = zeros(I,1);
    D = size(X,1) - 1;
    xi = zeros(D+1,K);
    
    for k = 1 : K
        a = a - phi_0;
        
        options = optimset('GradObj','on');
        initial_x = ones(2+D+1, 1);
        x = fminunc(@(x) fit_inclr_cost(x, X, w, a), initial_x, options);

        phi_0 = x(1);
        phi(k) = x(2);
        xi(:,k) = x(3:end);
        
        for i = 1 : I
            f = atan(xi(:,k)' * X(:,i));
            a(i) = a(i) + phi_0 + phi(k) * f;
        end  
    end
    
    I_test = size(X_test,2);
    predictions = zeros(1,I_test);
    for i = 1 : I_test
        act = phi_0;
        for k = 1 : K
            f = atan(xi(:,k)' * X_test(:,i));
            act = act + phi(k) * f;
        end
        predictions(i) = sigmoid(act);
    end
end