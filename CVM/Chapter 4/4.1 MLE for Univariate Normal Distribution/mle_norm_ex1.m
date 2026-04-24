% Author: Stefan Stavrev 2013

% Generate random values from the normal distribution with
% mean value original_mu and standard deviation original_sig.
original_mu = 5;
original_sig = 8;
r = original_mu + original_sig .* randn(100,1);

% Estimate the mean and the variance for the data in r.
[estimated_mu, estimated_var] = mle_norm(r);
estimated_sig = sqrt(estimated_var);

% Estimate and print the error for the mean and the standard deviation.
muError = abs(original_mu - estimated_mu);
sigError = abs(original_sig - estimated_sig);
disp([muError, sigError]);

% Plot the original and the estimated models for comparison.
x = -20:0.01:30;
original = normpdf(x, original_mu, original_sig);
estimated = normpdf(x, estimated_mu, estimated_sig);
plot(x, original, 'g', x, estimated, 'b');
xlabel('x', 'FontSize', 16);
ylabel('P(x)', 'FontSize', 16);
legend('original', 'estimated');
set(gca,'XTick',[],'YTick',[]);