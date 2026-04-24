% Author: Stefan Stavrev 2013

% Sample.
Phi = {@(x) mvnpdf(x,[0 0],[2 0; 0 2])};
S = {[1 2]};
D = 2;
Values = -5:.5:5;
T = 5000;
Samples = sample_gibbs (Phi, S, D, Values, T);
discard = 500; % discard first samples
numberOfSamples = (T-discard) / 5; % take every 5-th sample
Samples = Samples(:,discard+1:end);
Samples = Samples(:,1:5:end);

% Plot 2-D histogram.
K = length(Values);
H = zeros(K,K);
I = size(Samples,2);
Samples(2,:) = K + 1 - Samples(2,:);
for i = 1 : I
    col = Samples(1,i);
    row = Samples(2,i);
    H(row,col) = H(row,col) + 1;
end
imagesc(H);