% Author: Stefan Stavrev 2013

% Generate 2D data from normal distribution.
mu = [4 4];
sig = [1 0; 0 1];
X = mvnrnd(mu, sig, 20)';

% Create matrix XX that contains indices for the values in X.
% The function sample_gibbs_one uses indices of values, not the
% values directly.
XX = X;
Values = 0:0.5:10; % values that each variable (x and y) can take
value_index_map = containers.Map(Values, 1:length(Values));
for i = 1 : 20
    for j = 1 : 2
        % Get original value from X.
        temp = X(j,i);
        % Snap temp to the closest number in Values.
        m = mod(temp, 0.5);
        fl = floor(temp/0.5);
        if(m > 0.25)
            temp = (fl+1) * 0.5;
        else
            temp = fl * 0.5;
        end
        % Map value to index.
        XX(j,i) = value_index_map(temp);
    end
end

% Create symbolic function phi_1 and its gradient.
syms a b c d;
phi_1_sym = symfun(mvnpdf([a b], [c d], [2 0; 0 2]),[a b c d]);
phi_1_grad = gradient(phi_1_sym, [c d]);

% Learn Theta, the mean of the data.
I = size(X,2);
Samples = XX;
Theta = [0; 0]; % [mu_x; mu_y]
S = {[1 2]};
D = 2;
alpha = 0.1;
iterations = 0;
while true
    % Take one Gibbs sample step from each data point.
    for i = 1 : I
        Phi = {@(x) mvnpdf(x,[Theta(1) Theta(2)],[1 0; 0 1])};
        Samples(:,i) = sample_gibbs_one(XX(:,i), Phi, S, D, Values);
    end
    
    % Compute the approximate gradient.
    grad = 0;
    mu_x = Theta(1);
    mu_y = Theta(2);
    for i = 1 : I
        current_sample = Values([Samples(1,i) Samples(2,i)])';
        
        % derivative(log(f(x))) = derivative(f(x)) / f(x) was used to
        % simplify both terms in the approximate gradient.
        
        f_xi = eval(phi_1_sym(X(1,i), X(2,i), mu_x, mu_y));
        f_xi_star = eval(phi_1_sym(...
            current_sample(1), current_sample(2), mu_x, mu_y));
        
        grad_xi = eval(phi_1_grad(X(1,i), X(2,i), mu_x, mu_y));
        grad_xi_star = eval(phi_1_grad(...
            current_sample(1), current_sample(2), mu_x, mu_y));        
        
        temp = grad_xi / f_xi - grad_xi_star / f_xi_star;        
        grad = grad + temp;
    end
    
    % Update Theta with gradient descent.
    Theta = Theta + alpha * grad;
    
    iterations = iterations + 1;
    if (iterations == 5)
        break;
    end
end

% Plot the data points.
figure;
scatter(X(1,:), X(2,:), 60, 'fill',...
    'MarkerEdgeColor','k','MarkerFaceColor','b','LineWidth',1);
set(gca,'YDir','normal');
axis([0,10,0,10]);
set(gca,'YTick',[0,4,10], 'XTick',[0,4,10], 'FontSize', 16);
hold on;

% Visually compare the learned and the true mean.
plot(4, 4, 'o', 'MarkerSize', 15, 'MarkerFaceColor','g',...
    'MarkerEdgeColor','k');
hold on;
plot(Theta(1), Theta(2), 'o', 'MarkerSize', 15, 'MarkerFaceColor','r',...
    'MarkerEdgeColor','k');
hold on;
plot([4 4], [0 4], '--g');
hold on;
plot([0 4], [4 4], '--g');
hold on;
legend('Data','True Mean','Learned Mean');
hold off;