% Author: Stefan Stavrev 2013

% Create a grid. Set granularity to 100/1000, for low/high quality plots.
granularity = 100; 
a = -5;
b = 5;
domain = linspace (a, b, granularity);
[X,Y] = meshgrid (domain, domain);

% Generate 2D data from normal distributions.
mu = [-3 2;      
      0 -3;      
      4 3];
sig = [0.5 0; 0 0.5];
X_data = [mvnrnd(mu(1,:), sig, 5);          
          mvnrnd(mu(2,:), sig, 5);          
          mvnrnd(mu(3,:), sig, 5);];
 
% Prepare the training input.
X_train = [ones(1,size(X_data,1)); X_data(:,1)'];
w = X_data(:,2);
var_prior = 6;
X_test = [ones(1,granularity); domain];

% Train 6 Gaussian process regression models for different values for nu.
kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, 0.5);
[mu_test1, var_test1] = fit_gpr (X_train, w, var_prior, X_test, kernel);

kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, 0.7);
[mu_test2, var_test2] = fit_gpr (X_train, w, var_prior, X_test, kernel);

kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, 0.9);
[mu_test3, var_test3] = fit_gpr (X_train, w, var_prior, X_test, kernel);

kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, 1.5);
[mu_test4, var_test4] = fit_gpr (X_train, w, var_prior, X_test, kernel);

kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, 2);
[mu_test5, var_test5] = fit_gpr (X_train, w, var_prior, X_test, kernel);

kernel = @(x_i, x_j) kernel_gauss (x_i, x_j, 3);
[mu_test6, var_test6] = fit_gpr (X_train, w, var_prior, X_test, kernel);

mu_test = [mu_test1 mu_test2 mu_test3 mu_test4 mu_test5 mu_test6];
var_test = [var_test1 var_test2 var_test3 var_test4 var_test5 var_test6];

% Plot the results.
for m=1:6
    figure;
    Z = zeros(granularity,granularity);
    for j = 1 : granularity
        mu = mu_test(j,m);
        for i = 1 : granularity
            ww = domain(i);
            Z(i,j) = normpdf(ww,mu,var_test(j,m));
        end
    end
    pcolor(X,Y,Z);
    shading interp;
    colormap(hot(4096));
    set(gca,'YDir','normal');
    axis([a,b,a,b]);
    set(gca,'YTick',[a,b], 'XTick',[a,b], 'FontSize', 14);
    
    % Plot the data points on top.
    hold on;
    scatter(X_data(:,1), X_data(:,2), 120, 'fill','MarkerEdgeColor','w', ...
           'MarkerFaceColor','b','LineWidth',2);
    hold off;
end