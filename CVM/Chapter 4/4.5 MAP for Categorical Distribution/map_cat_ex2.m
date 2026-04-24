% Author: Stefan Stavrev 2013

% Generate random values from the categorical distribution
% with 6 categories and the corresponding probabilities.
original_probabilities = [0.25,0.15,0.1,0.1,0.15,0.25];
r1 = randsample (6, 200, true, original_probabilities);
r2 = randsample (6, 2000, true, original_probabilities);
r3 = randsample (6, 200000, true, original_probabilities);

% MAP estimate of the categorical distribution parameters from the data.
prior = [10 100 1000 1000 100 10];
estimated_probabilities_map1 = map_cat (r1, prior);
estimated_probabilities_map2 = map_cat (r2, prior);
estimated_probabilities_map3 = map_cat (r3, prior);

% Plot the original and the estimated models for comparison.
subplot(1,4,1);
bar(original_probabilities, 'b');
axis([0,7,0,0.5]);
xlabel('\lambda', 'FontSize', 16);
ylabel('P(\lambda)', 'FontSize', 16);
set(gca,'YTick',[0, 0.5]);
title('(a)', 'FontSize', 14);

subplot(1,4,2);
bar(estimated_probabilities_map1, 'r');
axis([0,7,0,0.5]);
xlabel('\lambda', 'FontSize', 16);
set(gca,'YTick',[0, 0.5]);
title('(b)', 'FontSize', 14);

subplot(1,4,3);
bar(estimated_probabilities_map2, 'r');
axis([0,7,0,0.5]);
xlabel('\lambda', 'FontSize', 16);
set(gca,'YTick',[0, 0.5]);
title('(c)', 'FontSize', 14);

subplot(1,4,4);
bar(estimated_probabilities_map3, 'r');
axis([0,7,0,0.5]);
xlabel('\lambda', 'FontSize', 16);
set(gca,'YTick',[0, 0.5]);
title('(d)', 'FontSize', 14);