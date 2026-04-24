% Author: Stefan Stavrev 2013

% Generate data from one Gaussian distribution.
N = 100;
data_mu = [1 2];
data_sig = [2 0; 0 .5];
x = mvnrnd (data_mu, data_sig, N);

% Generate few outliers.
N_outliers = 2;
outliers_mu = [-4 7];
outliers_sig = [0.2 0; 0 0.2];
outliers = mvnrnd (outliers_mu, outliers_sig, N_outliers);
x_plus_outliers = [x; outliers];

% Fit a Gaussian to the original data and to the data with outliers.
[lambda1, mu1, sig1] = fit_mog (x, 1, 0.01);
[lambda2, mu2, sig2] = fit_mog (x_plus_outliers, 1, 0.01);

% Fit a t-distribution to the data with outliers.
[t_mu, t_sig, t_nu] = fit_t (x_plus_outliers, 0.01);

% Create the common grid over which functions will be evaluated.
[XX,YY] = meshgrid (-10:0.1:10,-10:0.1:10);
xx = XX(:);
yy = YY(:);
x_y_matrix = [xx yy];
n = size(XX,1);

% Plot the data without outliers and the fitted Gaussian.
subplot(1,3,1);
scatter(x(:,1), x(:,2), 10, 'o');
axis([-6,6,-1,8]);
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca,'YTick',[-1,8], 'XTick',[-6,6]);
title('(a)', 'FontSize', 14);
% Plot the fitted Gaussian.
temp = mvnpdf (x_y_matrix, mu1(1,:), sig1{1});
gaussian1 = reshape (temp, n, n);
hold on;
contour(XX,YY,gaussian1);
hold off;

% Plot the data with outliers and the fitted Gaussian.
subplot(1,3,2);
scatter(x_plus_outliers(:,1), x_plus_outliers(:,2), 10, 'o');
axis([-6,6,-1,8]);
xlabel('x', 'FontSize', 16);
ylabel('', 'FontSize', 16);
set(gca,'YTick',[-1,8], 'XTick',[-6,6]);
title('(b)', 'FontSize', 14);
% Plot the fitted Gaussian.
temp = mvnpdf (x_y_matrix, mu2(1,:), sig2{1});
gaussian2 = reshape (temp, n, n);
hold on;
contour(XX,YY,gaussian2);
hold off;

% Plot the data with outliers and the fitted t-distribution.
subplot(1,3,3);
scatter(x_plus_outliers(:,1), x_plus_outliers(:,2), 10, 'o');
axis([-6,6,-1,8]);
xlabel('x', 'FontSize', 16);
ylabel('', 'FontSize', 16);
set(gca,'YTick',[-1,8], 'XTick',[-6,6]);
title('(c)', 'FontSize', 14);
% Plot the fitted t-distribution.
t_distribution = pdf_tm(x_y_matrix, t_mu, t_sig, t_nu);
t_distribution = reshape (t_distribution, n, n);
hold on;
contour(XX,YY,t_distribution);
hold off;