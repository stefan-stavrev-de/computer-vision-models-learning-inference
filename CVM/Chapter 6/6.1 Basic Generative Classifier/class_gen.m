% Author: Stefan Stavrev 2013

% Description: Classification based on multivariate measurement vector.
% Input: x_train - matrix of size (#training examples X dimensionality+1).
%                  The number of columns is the dimensionality of the data
%                  plus 1 for the last column, which contains class
%                  assignments for the corresponding row vectors.
%        x_test  - matrix of size (#test examples X dimensionality). The
%                  number of columns is one less than the number of columns
%                  in x_train, because there is no column for class
%                  assignments.
%        K       - number of classes. The classes throughout the code
%                  are assumed to be represented by integers 1:K.
% Output: lambda - categorical prior with K parameters, one for each
%                  of the K classes/categories.
%         mu     - matrix of size (K X dimensionality). Each row indexed by
%                  k, is the mean of the training examples in class k.
%         sig    - cell of K covariance matrices, where sig{k} is the
%                  covariance matrix for the class k.
%         posterior - matrix of size (size(x_test,1) X K), where each
%                  row is a posterior probability distribution over the
%                  K classes, for the corresponding row vector from the
%                  test data.
function [lambda, mu, sig, posterior] = class_gen (x_train, x_test, K)
    % Separate the training data based on the values in the last column,
    % which represents the class. The result is one matrix of training
    % data, per class. Some matrices may be empty, if the corresponding
    % class was not observed in the data. At the same time, compute
    % the counts for each class, that is how  many training examples belong
    % to each class.
    I = size (x_train, 1);
    dimensionality = size (x_train, 2) - 1;
    x_train_per_class = cell (1, K);
    class_counts = zeros (1, K);
    for i = 1 : I
        k = x_train (i, end);
        current_row = x_train (i, :);
        % Remove the last column.
        current_row (end) = [];
        x_train_per_class{k} = [x_train_per_class{k}; current_row];
        % Increase class count.
        class_counts(k) = class_counts(k) + 1;
    end
       
    % Compute mu, sigma and lambda for each class.
    mu = zeros (K, dimensionality);
    sig = cell (1, K);
    lambda = zeros (K, 1);
    for k = 1 : K
        % Compute mu.
        mu(k,:) = sum(x_train_per_class{k},1) ./ class_counts(k);
   
        % Compute sigma.
        sig{k} = zeros (dimensionality, dimensionality);
        for i = 1 : class_counts(k)
            mat = x_train_per_class{k}(i,:) - mu(k,:);
            mat = mat' * mat;
            sig{k} = sig{k} + mat;
        end
        sig{k} = sig{k} ./ class_counts(k);
        
        % Compute lambda.
        lambda(k) = class_counts(k) / I;
    end
    
    % Compute likelihoods for each class for the test data.
    likelihoods = zeros (size(x_test,1), K);
    for k = 1 : K
        likelihoods(:,k) = mvnpdf (x_test, mu(k,:), sig{k});
    end

    % Classify the data with Bayes' rule.
    denominator = 1 ./ (likelihoods * lambda);
    posterior = (likelihoods * diag(lambda));
    posterior = (posterior' * diag(denominator))';
end