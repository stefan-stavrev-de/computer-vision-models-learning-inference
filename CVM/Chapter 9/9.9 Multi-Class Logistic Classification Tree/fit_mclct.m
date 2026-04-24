% Author: Stefan Stavrev 2013

% Description: Multi-class logistic classification tree.
% Input: X - a (D+1)xI data matrix, where D is the data dimensionality
%            and I is the number of training examples. The training
%            examples are in X's columns rather than in the rows, and
%            the first row of X equals ones(1,I),
%        w - a Ix1 vector containing the corresponding world states for
%            each training example,
%        X_test - a data matrix containing training examples for which
%                 we need to make predictions,
%        J - wanted number of nodes in the tree,
%        G - linear functions in columns used as classifiers,
%        K - number of classes.
% Output: Predictions - (num_classes)xI_test matrix which contains the
%                       predicted class values for the data in X_test.
function Predictions = fit_mclct (X, w, X_test, J, G, K)
    I = size(X,2);
    I_test = size(X_test,2);
    M = size(G,2);
    c = zeros(J,1);
    Lambda = zeros(K,J+1);
    GX = sigmoid(G' * X) > 0.5;
    GX_test = sigmoid(G' * X_test) > 0.5;
    Predictions = zeros(K,I_test);
        
    % Init queue.
    import java.util.LinkedList;
    queue = LinkedList();
    add(queue, 1:I);
    
    % For each node in the tree.
    for j = 1 : J
        current_data = remove(queue);
        II = length(current_data);
        l = zeros(M,1);
        for m = 1 : M
            % Count frequency for k-th class in left and right branches.
            n_l = zeros(K,1);
            n_r = zeros(K,1);
            for i = 1 : II
                ii = current_data(i);
                k = w(ii);
                if(GX(m,ii) == 1)
                    n_r(k) = n_r(k) + 1;
                else
                    n_l(k) = n_l(k) + 1;
                end
            end
            
            % Compute log likelihood.
            sum_l = sum(n_l);
            sum_r = sum(n_r);
            % If this classifier can not separate the data then it should
            % not be applied.
            if(sum_l == 0 || sum_r == 0)
                l(m) = Inf;
            else
                % Protect against zero values.
                n_l = n_l + 1;
                n_r = n_r + 1;
                sum_l = sum_l + K;
                sum_r = sum_r + K;
                
                % Compute log likelihood.
                norm_l = log(n_l ./ sum_l);
                norm_r = log(n_r ./ sum_r);
                l(m) = sum(norm_l) + sum(norm_r);
            end            
        end
        % Store index of best classifier for this node.
        [~, c(j)] = min(l);
        % Partition in two sets.
        right = GX(c(j),current_data);
        add(queue, current_data(not(right)));
        add(queue, current_data(right));
    end
    
    % Recover categorical parameters at J+1 leaves.
    for p = 1 : J + 1
        current_data = remove(queue);
        II = length(current_data);
        n = zeros(K,1);
        for i = 1 : II
            ii = current_data(i);
            k = w(ii);
            n(k) = n(k) + 1;
        end
        Lambda(:,p) = n ./ sum(n);
    end
    
    % Predict by moving each test example from the root to the bottom
    % of the tree, and then taking the corresponding categorical
    % distribution from the leaf node.
    for i = 1 : I_test
        j = 1;        
        right = 0;
        tree_level = 0;
        while true
            right = GX_test(c(j),i);
            new_j = 2*j;
            if (right == true)
                new_j = new_j + 1;
            end
            if(new_j > J)
                break;
            else
                j = new_j;
                tree_level = tree_level + 1;
            end            
        end
        node_index_in_last_level_left_to_right = j - 2^tree_level;
        if (right == false)
            leaf_node = node_index_in_last_level_left_to_right * 2 + 1;            
        else
            leaf_node = node_index_in_last_level_left_to_right * 2 + 2;
        end
        Predictions(:,i) = Lambda(:,leaf_node);
    end
end