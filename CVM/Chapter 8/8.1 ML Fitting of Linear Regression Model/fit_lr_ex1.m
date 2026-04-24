% Author: Stefan Stavrev 2013

% Generate data points.
I = 50;
x = linspace(1,9,I);
phi = [7; -0.5];
sig = 0.6;
w = zeros(I,1);
X = [ones(I,1) x'];
X_phi = X * phi;
r = sig * randn(I,1);
for i = 1 : I
    mu = X_phi(i);
    w(i) = mu + r(i);
end

% Fit a linear regression model.
[fit_phi, fit_sig] = fit_lr (X', w);

% Plot results.
granularity = 2000;
domain = linspace(0,10,granularity);
[X,Y]=meshgrid(domain,domain);
XX = [ones(granularity,1) domain'];
temp = XX * fit_phi;
Z = zeros(granularity,granularity);
for j = 1 : granularity
    mu = temp(j);
    for i = 1 : granularity
        ww = domain(i);
        Z(i,j) = normpdf(ww,mu,fit_sig);
    end
end
pcolor(X,Y,Z);
shading interp;
colormap(hot(4096));
set(gca,'YDir','normal');
axis([0,10,0,10]);
xlabel('x', 'FontSize', 16);
ylabel('w', 'FontSize', 16);
set(gca,'YTick',[0,10], 'XTick',[0,10]);

% Plot the data points on top of the heat map.
hold on;
scatter(x,w,120,'fill','MarkerEdgeColor','w','MarkerFaceColor','b',...
        'LineWidth',2);
hold off;