% Author: Stefan Stavrev 2013

% Create a grid. Set granularity to 100/1000, for low/high quality plots.
granularity = 100; 
a = -5;
b = 5;
domain = linspace (a, b, granularity);
[X,Y] = meshgrid (domain, domain);
x = X(:);
y = Y(:);
n = size(X,1);

% Generate 2D data from normal distributions.
mu = [-1 2.5;
      1 -2.5];
sig = [0.5 0; 0 0.5];
points_per_class = 20;
X_data = [mvnrnd(mu(1,:), sig, points_per_class);          
          mvnrnd(mu(2,:), sig, points_per_class);];
      
% Prepare the training input.
X_train = [ones(1,size(X_data,1)); X_data'];
w = [zeros(points_per_class,1); ones(points_per_class,1)];
var_prior = 6;
X_test = [ones(1,granularity*granularity); x'; y'];

lambda = 0.3;

% Fit a relevance vector classification model.
initial_psi = zeros(size(X_train,2), 1);
nu = 0.0005;
[predictions, relevant_points] = fit_rvc (X_train, w, nu, X_test, initial_psi, @kernel_gauss, lambda);

% Plot the results.
figure;
Z = reshape(predictions,n,n);
pcolor(X,Y,Z);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b], 'FontSize', 16);
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);

selector0 = relevant_points;
selector1 = relevant_points;

selector0(points_per_class+1 : 2*points_per_class) = 0;
selector1(1 : points_per_class) = 0;

selector0n = selector0;
selector0n(1 : points_per_class) = not(selector0n(1 : points_per_class));

selector1n = selector1;
selector1n(points_per_class+1 : 2*points_per_class) = not(selector1n(points_per_class+1 : 2*points_per_class));

% Plot relevant points in class 0.
hold on;
scatter(X_data(selector0,1), X_data(selector0,2), 200, 'fill',...
       'MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1);
   
% Plot non-relevant points in class 0.
hold on;
scatter(X_data(selector0n,1), X_data(selector0n,2), 50, 'fill',...
       'MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1);

% Plot relevant points in class 1.
hold on;
scatter(X_data(selector1,1), X_data(selector1,2), 200, 'fill',...
       'MarkerEdgeColor','k','MarkerFaceColor','g','LineWidth',1);
   
% Plot non-relevant points in class 1.
hold on;
scatter(X_data(selector1n,1), X_data(selector1n,2), 50, 'fill',...
       'MarkerEdgeColor','k','MarkerFaceColor','g','LineWidth',1);
  
hold off;