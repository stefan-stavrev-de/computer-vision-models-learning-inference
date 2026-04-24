% Author: Stefan Stavrev 2013

% Create a grid. Set granularity to 100/1000, for low/high quality plots.
granularity = 1000; 
a = -5;
b = 5;
domain = linspace (a, b, granularity);
[X,Y] = meshgrid (domain, domain);
x = X(:);
y = Y(:);
n = size(X,1);

% Generate 2D data from normal distributions.
mu = [-.5 .5;
      .5 -.5];
sig = [2 0; 0 2];
points_per_class = 20;
X_data = [mvnrnd(mu(1,:), sig, points_per_class);          
          mvnrnd(mu(2,:), sig, points_per_class);];
      
% Prepare the training input.
X_train = [ones(1,size(X_data,1)); X_data'];
w = [zeros(points_per_class,1); ones(points_per_class,1)];
X_test = [ones(1,granularity*granularity); x'; y'];

% Construct the weak classifiers' parameters.
M = 20 * 40; % 20 angles, for each angle 40 offsets
Alpha = zeros(3,M);
angle_delta = 2*pi / 20;
column = 1;
offset_delta = 1 / 40;
angle = 0;
for i = 1 : 20
    x = cos(angle);
    y = sin(angle);
    offset = 0;
    for j = 1 : 40    
        Alpha(1,column) = offset;
        Alpha(2,column) = x;
        Alpha(3,column) = y;
        offset = offset + offset_delta;
        column = column + 1;
    end
    angle = angle + angle_delta;
end

% Fit a logitboost model.
predictions = fit_logitboost (X_train, w, X_test, Alpha);

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