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
mu = [2 0;
    -2 0];
sig = [.5 0; 0 .5];
points_per_class = 10;
X_data = [mvnrnd(mu(1,:), sig, points_per_class);
    mvnrnd(mu(2,:), sig, points_per_class);];

% Prepare the training input.
X_train = [ones(1,size(X_data,1)); X_data'];
w = [zeros(points_per_class,1); ones(points_per_class,1)] + 1;
X_test = [ones(1,granularity*granularity); x'; y'];

% Construct the classifiers' parameters.
G = [0 0 0 0;
    1 0 cosd(45) cosd(135);
    0 1 sind(45) sind(135)];

% Fit a logitboost model.
J = 1;
K = 2;
Predictions = fit_mclct(X_train, w, X_test, J, G, K);
Predictions = Predictions(2,:);

% Plot the results.
figure;
Z = reshape(Predictions,n,n);
pcolor(X,Y,Z);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([a,b,a,b]);
set(gca,'YTick',[a,0,b], 'XTick',[a,0,b], 'FontSize', 16);
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);

% Plot the data points on top.
hold on;
selector = 1:points_per_class;
scatter(X_data(selector,1), X_data(selector,2), 100, 'fill',...
    'MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1);

hold on;
selector = points_per_class+1 : 2*points_per_class;
scatter(X_data(selector,1), X_data(selector,2), 100, 'fill',...
    'MarkerEdgeColor','k','MarkerFaceColor','g','LineWidth',1);

hold off;