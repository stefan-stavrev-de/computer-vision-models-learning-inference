% Author: Stefan Stavrev 2013

% Vary alpha only and plot results.
x = 0:0.01:10;
subplot(1,2,1);
px1 = pdf_gamma (x, 5, 4);
px2 = pdf_gamma (x, 10, 4);
px3 = pdf_gamma (x, 20, 4);
px4 = pdf_gamma (x, 30, 4);
plot (x,px1,'b', x,px2,'g', x,px3,'y', x,px4,'r', 'LineWidth',2);
xlabel('x', 'FontSize', 16);
ylabel('P(x)', 'FontSize', 16);
legend('\alpha=5,   \beta=4', '\alpha=10, \beta=4', ...
       '\alpha=20, \beta=4','\alpha=30, \beta=4');
axis([0,10,0,1]);
set(gca,'XTick',[0,5,10],'YTick',[0,1]);
title('(a)', 'FontSize', 14);

% Vary beta only and plot results.
subplot(1,2,2);
x = 0:0.01:30;
px1 = pdf_gamma (x, 20, 8);
px2 = pdf_gamma (x, 20, 4);
px3 = pdf_gamma (x, 20, 2);
px4 = pdf_gamma (x, 20, 1);
plot (x,px1,'b', x,px2,'g', x,px3,'y', x,px4,'r', 'LineWidth',2);
xlabel('x', 'FontSize', 16);
ylabel('', 'FontSize', 16);
legend('\alpha=20, \beta=8', '\alpha=20, \beta=4', ...
       '\alpha=20, \beta=2','\alpha=20, \beta=1');
axis([0,30,0,1]);
set(gca,'XTick',[0,10,20,30],'YTick',[0,1]);
title('(b)', 'FontSize', 14);