% Author: Stefan Stavrev 2013

% Create a grid.
granularity = 20;
a = -5;
b = 5;
domain = linspace (a, b, granularity);
[X,Y] = meshgrid (domain, domain);
x = X(:);
y = Y(:);
n = size(X,1);

% Generate data ----------------------------------------------------------

% Generate points on a plane.
x_y_matrix = [ones(n*n,1) x y];
phi = [0; 0.8; 5]; % plane equation coefficients
Z = x_y_matrix * phi; % evaluate the plane function

% Offset the points a bit from the plane.
offset = 0.8*randn(size(Z));
X_points = x + offset;
Y_points = y + offset;
Z_points = Z + offset;

% Remove points near the borders.
selector_1 = X_points < 4.5 & X_points > -4.5;
selector_2 = Y_points < 4.5 & Y_points > -4.5;
selector = selector_1 & selector_2;
X_points = X_points(selector);
Y_points = Y_points(selector);
Z_points = Z_points(selector);

% Decrease the number of points.
selector = randperm(length(X_points)); % generate random integers
selector = selector(1:30);
X_points = X_points(selector);
Y_points = Y_points(selector);
Z_points = Z_points(selector);

% Prepare the training data.
I = length(X_points);
X_train = [ones(1,I); X_points'; Y_points'];
w = Z_points;
X_test = [ones(1,granularity*granularity); x'; y'];
 
 % ------------------------------------------------------------------------

% Fit Bayesian linear regression model.
var_prior = 6;
[mu_test2, var_test] = fit_blr (X_train, w, var_prior, X_test);

% Plot the Bayesian linear regression fit ---------------------------------

% Plot the predicted values.
figure;
ZZ = reshape(mu_test2,n,n);
pcolor(X,Y,ZZ);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('', 'FontSize', 12);
ylabel('', 'FontSize', 12);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b]);
title('', 'FontSize', 14);

% Plot the training data points on top.
hold on;
colormap(hot(4096));
scatter(X_points,Y_points,40,Z_points,'fill','MarkerEdgeColor','k');
hold off;

% -------------------------------------------------------------------------

% Fit sparse linear regression model.
nu = 0.0005;
mu_test2 = fit_slr (X_train, w, nu, X_test); % we only need mean values

% Plot the sparse linear regression fit -----------------------------------

% Plot the predicted values.
figure;
ZZ = reshape(mu_test2,n,n);
pcolor(X,Y,ZZ);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
xlabel('', 'FontSize', 12);
ylabel('', 'FontSize', 12);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b]);
title('', 'FontSize', 14);

% Plot the training data points on top.
hold on;
colormap(hot(4096));
scatter(X_points,Y_points,40,Z_points,'fill','MarkerEdgeColor','k');
hold off;

% -------------------------------------------------------------------------