% Author: Stefan Stavrev 2013

% Generate random values from the categorical distribution
% with 6 categories and the corresponding probabilities.
original_probabilities = [0.25,0.15,0.1,0.1,0.15,0.25];
r = randsample (6, 1000000, true, original_probabilities);

% Bayesian approach.
prior = [1 1 1 1 1 1];
[alpha_post, prediction] = by_cat (r, prior);

% MAP estimate for comparison purposes.
estimated_probabilities_map = map_cat (r, prior);

% Plot the original and the estimated models for comparison.
subplot(1,3,1);
bar (original_probabilities, 'g');
axis([0,7,0,0.4]);
xlabel('\lambda', 'FontSize', 16);
ylabel('P(\lambda)', 'FontSize', 16);
set(gca,'YTick',[0, 0.4]);
title('(d)', 'FontSize', 14);

subplot(1,3,2);
bar (estimated_probabilities_map, 'b');
axis([0,7,0,0.4]);
xlabel('\lambda', 'FontSize', 16);
ylabel('', 'FontSize', 16);
set(gca,'YTick',[0, 0.4]);
title('(e)', 'FontSize', 14);

subplot(1,3,3);
bar (prediction, 'r');
axis([0,7,0,0.4]);
xlabel('\lambda', 'FontSize', 16);
ylabel('', 'FontSize', 16);
set(gca,'YTick',[0, 0.4]);
title('(f)', 'FontSize', 14);