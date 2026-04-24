% Author: Stefan Stavrev 2013

% Description: return true if the matrix M is diagonal.
function [diagonal] = isdiag (M)
    [m, n] = size (M);
    if m ~= n
        diagonal = false;
        return;
    end
    diagonal = true;
    for i = 1 : n
        for j = 1 : n
            if i~=j && M(i,j)~=0
                diagonal = false;
                return;
            end
        end
    end
end