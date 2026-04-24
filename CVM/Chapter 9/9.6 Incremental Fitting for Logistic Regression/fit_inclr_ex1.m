% Author: Stefan Stavrev 2013

figure;

% Generate data for class 0.
I_0 = 20;
mu_0 = -2;
sigma_0 = 1.5;
class_0 = normrnd(mu_0, sigma_0, 1, I_0);
scatter(class_0, zeros(1,I_0),100,'fill','MarkerEdgeColor','k', ...
       'MarkerFaceColor','b','LineWidth',1);
hold on;

% Generate data for class 1.
I_1 = 20;
mu_1 = 2;
sigma_1 = 1.5;
class_1 = normrnd(mu_1, sigma_1, 1, I_1);
scatter(class_1, zeros(1,I_1),100,'fill','MarkerEdgeColor','k', ...
       'MarkerFaceColor','g','LineWidth',1);
   
% Plot properties.
xlabel('x', 'FontSize', 16);
ylabel('', 'FontSize', 16);
axis([-5,5,-0.1,1.1]);
set(gca,'XTick',[-5,0,5],'YTick',[0,1],'FontSize', 16);
title('(a)', 'FontSize', 16);

% Prepare training data.
X = [class_0 class_1];
X = [ones(1,I_0+I_1); X];
w = [zeros(I_0,1); ones(I_1,1)];
X_test = -5:0.1:5;
X_test = [ones(1,size(X_test,2)); X_test];

% Fit an iterative logistic regression model.
predictions = fit_inclr (X, w, X_test);

% Plot the predictions.
plot(-5:0.1:5, predictions, 'LineWidth',2,'LineSmoothing','on','Color','r');
hold off;