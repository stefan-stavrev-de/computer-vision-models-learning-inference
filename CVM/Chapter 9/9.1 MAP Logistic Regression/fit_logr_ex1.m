% Author: Stefan Stavrev 2013

figure;
x = -6:0.1:6;

subplot(2,2,1);
y1 = -4 * x;
plot(x,y1,'LineWidth',2,'LineSmoothing','on','Color','r');
hold on;

y2 = -2 * x;
plot(x,y2,'LineWidth',2,'LineSmoothing','on','Color','g');

y3 = 2 * x;
plot(x,y3,'LineWidth',2,'LineSmoothing','on','Color','y');

y4 = 4 * x;
plot(x,y4,'LineWidth',2,'LineSmoothing','on','Color','b');
hold off;

% Plot settings.
xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
axis([-2,2,-5,5]);
set(gca,'XTick',[-2,0,2],'YTick',[-5,0,5],'FontSize', 14);
title('(a) y = mx', 'FontSize', 16);

subplot(2,2,2);
y5 = sigmoid(y1);
plot(x,y5,'LineWidth',2,'LineSmoothing','on','Color','r');
hold on;

y6 = sigmoid(y2);
plot(x,y6,'LineWidth',2,'LineSmoothing','on','Color','g');

y7 = sigmoid(y3);
plot(x,y7,'LineWidth',2,'LineSmoothing','on','Color','y');

y8 = sigmoid(y4);
plot(x,y8,'LineWidth',2,'LineSmoothing','on','Color','b');
hold off;

xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
axis([-3,3,-0.1,1.1]);
set(gca,'XTick',[-3,0,3],'YTick',[0,0.5,1],'FontSize', 14);
title('(b) y = sig(mx)', 'FontSize', 16);

subplot(2,2,3);
y9 =  x;
plot(x,y9,'LineWidth',2,'LineSmoothing','on','Color','r');
hold on;

y10 =  x + 2;
plot(x,y10,'LineWidth',2,'LineSmoothing','on','Color','g');

y11 =  x - 2;
plot(x,y11,'LineWidth',2,'LineSmoothing','on','Color','b');
hold off;

xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
axis([-5,5,-5,5]);
set(gca,'XTick',[-5,0,5],'YTick',[-5,0,5],'FontSize', 14);
title('(c) y = x + b', 'FontSize', 16);

subplot(2,2,4);
y12 =  sigmoid(y9);
plot(x,y12,'LineWidth',2,'LineSmoothing','on','Color','r');
hold on;

y13 =  sigmoid(y10);
plot(x,y13,'LineWidth',2,'LineSmoothing','on','Color','g');

y14 =  sigmoid(y11);
plot(x,y14,'LineWidth',2,'LineSmoothing','on','Color','b');
hold off;

xlabel('x', 'FontSize', 16);
ylabel('y', 'FontSize', 16);
axis([-6,6,-0.1,1.1]);
set(gca,'XTick',[-6,0,6],'YTick',[0,0.5,1],'FontSize', 14);
title('(d) y = sig(x + b)', 'FontSize', 16);