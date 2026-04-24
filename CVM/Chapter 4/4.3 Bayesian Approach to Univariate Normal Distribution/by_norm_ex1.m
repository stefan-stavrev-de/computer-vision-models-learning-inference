% Author: Stefan Stavrev 2013

% Generate random values from the normal distribution with
% mean value original_mu and standard deviation original_sig.
original_mu = 5;
original_sig = 8;
I = 5;
r = original_mu + original_sig .* randn(I,1);

% Estimate the mean and the variance for the data in r.
% Values used for alpha, beta, gamma and delta are (1,1,1,0), for the
% sake of the example. Other values can be tried too.
x_test = -20:0.01:30;
[alpha_post, beta_post, gamma_post, delta_post, x_prediction] = ...
    by_norm(r, 1, 1, 1, 0, x_test);

% MAP, for comparison purposes to the Bayesian approach.
[map_mu, map_var] = map_norm(r, 1, 1, 1, 0);
map_sig = sqrt(map_var);

% Plot the original and the estimated models for comparison.
original = normpdf(x_test, original_mu, original_sig);
map = normpdf(x_test, map_mu, map_sig);
%plot(x_test, original, 'g', x_test, map, 'b', x_test, x_prediction, 'r');
semilogy(x_test, original, 'g', x_test, map, 'b', x_test, x_prediction, 'r');
xlabel('x', 'FontSize', 16);
ylabel('log (P(x))', 'FontSize', 16);
legend('Original', 'MAP', 'Bayesian');
axis([-20,30,0,0.1]);
set(gca,'XTick',[],'YTick',[]);
title('(c)', 'FontSize', 14);