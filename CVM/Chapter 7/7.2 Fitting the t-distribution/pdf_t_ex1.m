% Author: Stefan Stavrev 2013

% Plot t-distributions on linear scale.
subplot(1,2,1);
x = -5:0.01:5;
mu = 0;
sig = 1;
px1 = pdf_t (x, mu, sig, 1);
px2 = pdf_t (x, mu, sig, 2);
px3 = pdf_t (x, mu, sig, 5);
norm_pdf = normpdf (x, mu, sig);
plot (x,px1,'b', x,px2,'g', x,px3,'y', x,norm_pdf,'r', 'LineWidth',2);
xlabel('x', 'FontSize', 16);
ylabel('P(x)', 'FontSize', 16);
legend('t-distribution, \nu = 1', 't-distribution, \nu = 2', ...
       't-distribution, \nu = 5','normal distribution');
axis([-5,5,0,0.6]);
set(gca,'XTick',[-5,0,5],'YTick',[0,0.4]);
title('(a)', 'FontSize', 14);

% Plot t-distributions on log scale.
subplot(1,2,2);
semilogy (x,px1,'b', x,px2,'g', x,px3,'y', x,norm_pdf,'r', 'LineWidth',2);
xlabel('x', 'FontSize', 16);
ylabel('log(P(x))', 'FontSize', 16);
legend('t-distribution, \nu = 1', 't-distribution, \nu = 2', ...
       't-distribution, \nu = 5','normal distribution');
axis([-5,5,0,100]);
set(gca,'XTick',[-5,0,5],'YTick',[0,1]);
title('(b)', 'FontSize', 14);