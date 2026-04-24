% Author: Stefan Stavrev 2013

% Generate random values from the categorical distribution 
% with 6 categories and the corresponding probabilities.
original_probabilities = [0.25,0.15,0.1,0.1,0.15,0.25];
r = randsample (6, 100, true, original_probabilities);

% MAP estimate of the categorical distribution parameters from the data.
prior = [1 1 1 1 1 1];
estimated_probabilities_map = map_cat (r, prior);

% Compute the MLE estimate for comparison.
estimated_probabilities_mle = mle_cat (r, 6);

% Plot the original and the estimated models for comparison.
subplot(1,3,1);
bar(original_probabilities, 'b');
axis([0,7,0,0.4]);
xlabel('\lambda', 'FontSize', 16);
ylabel('P(\lambda)', 'FontSize', 16);
set(gca,'YTick',[0, 0.4]);
title('(a)', 'FontSize', 14);

subplot(1,3,2);
bar(estimated_probabilities_map, 'r');
axis([0,7,0,0.4]);
xlabel('\lambda', 'FontSize', 16);
set(gca,'YTick',[0, 0.4]);
title('(b)', 'FontSize', 14);

subplot(1,3,3);
bar(estimated_probabilities_mle, 'r');
axis([0,7,0,0.4]);
xlabel('\lambda', 'FontSize', 16);
set(gca,'YTick',[0, 0.4]);
title('(c)', 'FontSize', 14);