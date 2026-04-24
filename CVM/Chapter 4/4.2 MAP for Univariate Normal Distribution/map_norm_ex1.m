% Author: Stefan Stavrev 2013

% Generate random values from the normal distribution with
% mean value original_mu and standard deviation original_sig.
original_mu = 5;
original_sig = 8;
% I can be modified in order to see how MAP behaves for small vs big
% amounts of data.
I = 1000000;
r = original_mu + original_sig .* randn(I,1);

% Estimate the mean and the variance for the data in r.
% Values used for alpha, beta, gamma and delta are (1,1,1,0), for the
% sake of the example. Other values can be tried too.
[estimated_mu, estimated_var] = map_norm(r, 1, 1, 1, 0);
disp(['Estimated mean: ', num2str(estimated_mu)])
estimated_sig = sqrt(estimated_var);

[mle_mu, mle_var] = mle_norm(r);
mle_sig = sqrt(mle_var);

% Estimate and print the error for the mean and the standard deviation.
muError = abs(original_mu - estimated_mu);
sigError = abs(original_sig - estimated_sig);
disp(['Errors: ', num2str(muError), ', ', num2str(sigError)]);

% Plot the original and the estimated models for comparison.
x = -20:0.01:30;
original = normpdf(x, original_mu, original_sig);
estimated = normpdf(x, estimated_mu, estimated_sig);
mle = normpdf(x, mle_mu, mle_sig);
plot(x, original, 'g', x, estimated, 'b', x, mle, 'r');
xlabel('x', 'FontSize', 16);
ylabel('', 'FontSize', 16);
legend('original', 'MAP', 'MLE');
axis([-20,30,0,0.1]);
set(gca,'XTick',[],'YTick',[]);
title('(d)', 'FontSize', 14);