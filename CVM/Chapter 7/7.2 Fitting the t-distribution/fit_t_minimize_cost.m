% Author: Stefan Stavrev 2013

% The minimization method is called "Direct Line Search".
% Read more in Appendix B.3 in Dr. Prince's book.
function [nu] = fit_t_minimize_cost (E_hi, E_log_hi)
    % The initial range is [0,1000].
    a = 0;
    d = 1000;
    while true
        third = (d-a)/3;        
        b = a + third;
        c = d - third;
        %msg = [num2str(a) ' ' num2str(b) ' ' num2str(c) ' ' num2str(d)];
        %disp(msg);
        b_cost = fit_t_cost (b, E_hi, E_log_hi);
        c_cost = fit_t_cost (c, E_hi, E_log_hi);
        if b_cost < c_cost            
            d=c; % The new search range is [a,c].
        else
            a=b; % The new search range is [b,d].
        end
        
        if d-a < 1
            nu = d; % We could have picked a also.
            break;
        end
    end
end