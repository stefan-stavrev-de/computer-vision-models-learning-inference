% Author: Stefan Stavrev 2013

% Construct the input dataset from the original dataset.
I = 1000;
D = 60*60*3;
X = zeros(I, D);
for i=1:I
    img = faceNorm(:,:,:,i);
    X(i,:) = img(:);
end

% Fit a factor analyzer.
K = 10;
iterations = 10;
[mu, phi, sig] = fit_fa (X, K, iterations);

% Plot mu.
figure;
mu = mu ./ max(mu);
mu_mat = reshape(mu,60,60,3);
image(mu_mat);

% Plot sig.
figure;
sig = sig ./ max(sig);
sig_mat = reshape(sig,60,60,3);
image(sig_mat);

% Example similar to (Figure 7.22) in Dr. Prince's book.
rate = 0.001;
mu_phi_lin_combinations = zeros(8,D);
for i=1:4
    phi_ = phi(:,i)';
    
    % Go away from mu in positive phi_ direction.
    v = mu;
    while true
        new = v + rate*phi_;
        if sum(new<0)==0
            v=new;
        else
            break;        
        end    
    end
    mu_phi_lin_combinations(i,:) = v;
    
    % Go away from mu in negative phi_ direction.
    v = mu;
    while true
        new = v - rate*phi_;
        if sum(new<0)==0
            v=new;
        else
            break;        
        end    
    end
    mu_phi_lin_combinations(4+i,:) = v;
end

% Reshape and normalize.
mu_phi_lin_combinations_mat = cell (1, 8);
for i=1:8
    mp = mu_phi_lin_combinations(i,:);
    mp = mp ./ max(mp);
    mp = reshape(mp,60,60,3);    
    mu_phi_lin_combinations_mat{i} = mp;
end

% Plot the linear combinations between mu and the phi column vectors.
figure;
for i=1:8
    subplot(2,4,i);
    image(mu_phi_lin_combinations_mat{i});
end