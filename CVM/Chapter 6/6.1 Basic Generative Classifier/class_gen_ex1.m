% Author: Stefan Stavrev 2013

% Generate N random positive numbers, and N negative numbers.
% The mean is 2 and -2 respectively, the std is 0.5.
N = 100;
r1 = -2 + 0.5 .* randn(N,1);
r2 = 2 + 0.5 .* randn(N,1);
r = [r1; r2];
y = zeros(N, 1);

% Plot negative values in blue.
figure;
plot (r1,y,'bo');
hold on
plot (r2,y,'ro');

% Draw line x=0.
hold on
xx = linspace(0, 0, 10);
yy = linspace(-2, 2, 10);
plot(xx,yy,'k')

% Draw line y=0.
hold on
xx = linspace(-10, 10, 10);
yy = linspace(0, 0, 10);
plot(xx,yy,'k')

axis([-4,4,-0.1,0.1]);
xlabel('x', 'FontSize', 16);
set(gca,'YTick',[], 'XTick',[-4,0,4]);
title('(a)', 'FontSize', 14);
hold off

% Attach an additional column to the data for the class assignments.
% The classes are 1 and 2, instead of 0 and 1, just a convenience
% for better workflow in Matlab because indexing begins at 1.
class_column = [ones(N,1); ones(N,1)+1];
x_train = [r class_column];

% Make predictions for one negative and one positive value.
x_test = (-4:0.01:4)';
[lambda, mu, sig, posterior] = class_gen (x_train, x_test, 2);

% Plot the posterior for the two test examples in x_test.
figure;
plot(x_test, posterior(:,1), 'b', x_test, posterior(:,2), 'r');
axis([-4,4,0,1.5]);
xlabel('x', 'FontSize', 16);
ylabel('P(w|x)', 'FontSize', 16);
legend('P(w=1|x)', 'P(w=2|x)');
set(gca,'YTick',[0, 1], 'XTick',[-4,-2,0,2,4]);
title('(d)', 'FontSize', 14);

% Plot lambda which represents P(w).
figure;
bar(lambda, 'y');
axis([0.5,2.5,0,1]);
xlabel('w', 'FontSize', 16);
ylabel('P(w)', 'FontSize', 16);
set(gca,'YTick',[0, 0.5, 1]);
title('(b)', 'FontSize', 14);

% Plot mu and sig which represent P(x|w=0) and P(x|w=1).
figure;
x = -4:0.01:4;
norm_pdf_1 = normpdf (x, mu(1,:), sig{1});
norm_pdf_2 = normpdf (x, mu(2,:), sig{2});
plot(x, norm_pdf_1, 'b', x, norm_pdf_2, 'r');
axis([-4,4,0,2]);
xlabel('x', 'FontSize', 16);
ylabel('P(x|w)', 'FontSize', 16);
legend('P(x|w=1)', 'P(x|w=2)');
set(gca,'XTick',[-4,-2,0,2,4],'YTick',[0,1,2]);
title('(c)', 'FontSize', 14);