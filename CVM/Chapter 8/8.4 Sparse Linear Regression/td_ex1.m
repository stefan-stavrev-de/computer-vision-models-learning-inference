% Author: Stefan Stavrev 2013

% Create a grid.
granularity = 1000;
a = -3;
b = 3;
domain = linspace (a, b, granularity);
[X,Y] = meshgrid (domain, domain);
x = X(:);
y = Y(:);
x_y_matrix = [x y];
n = size(X,1);

% Define common parameters mu and sig.
mu = [0 0];
sig = eye(2);
nu = 0.01;

% Compute a product of two univariate t-distributions.
ptd = pdf_t (domain, 0, 1, nu);
ptd = ptd' * ptd;

% Plot the product of two univariate t-distributions.
subplot(1,2,1);
pcolor(X,Y,ptd);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b]);
title('(e)', 'FontSize', 14);

% Compute 2D multivariate t-distribution.
mtd = pdf_tm (x_y_matrix, mu, sig, nu);
mtd = reshape (mtd, n, n);

% Plot the 2D multivariate t-distribution.
subplot(1,2,2);
pcolor(X,Y,mtd);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('x', 'FontSize', 16);
ylabel('', 'FontSize', 16);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b]);
title('(f)', 'FontSize', 14);