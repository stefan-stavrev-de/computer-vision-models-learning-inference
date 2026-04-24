% Author: Stefan Stavrev 2013

U = [1 3; 2 2; 3 1];
P = {[], [2 1; 4 3], [3 2; 1 4]};
w = viterbi_chain(U, P);
disp(w);