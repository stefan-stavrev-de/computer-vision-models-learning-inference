% Author: Stefan Stavrev 2013

% Generate random values from the Dirichlet distribution with 6 parameters.
%r = randdir ([1 1 1 1 1 1], 3);
%r = randdir ([1000 1000 1000 1000 1000 1000], 3);
r = randdir ([10 100 1000 1000 100 10], 3);

% Plot the first sample.
subplot(1,3,1);
bar(r(1,:), 'y');
axis([0,7,0,0.6]);
xlabel('\lambda', 'FontSize', 16);
ylabel('P(\lambda)', 'FontSize', 16);
set(gca,'YTick',[0, 0.6]);
title('(g)', 'FontSize', 14);

% Plot the second sample.
subplot(1,3,2);
bar(r(2,:), 'b');
axis([0,7,0,0.6]);
xlabel('\lambda', 'FontSize', 16);
ylabel('', 'FontSize', 16);
set(gca,'YTick',[0, 0.6]);
title('(h)', 'FontSize', 14);

% Plot the third sample.
subplot(1,3,3);
bar(r(3,:), 'b');
axis([0,7,0,0.6]);
xlabel('\lambda', 'FontSize', 16);
ylabel('', 'FontSize', 16);
set(gca,'YTick',[0, 0.6]);
title('(i)', 'FontSize', 14);