function x = solve_framework(x, A, b, flag_pcg, tol_cg, maxit_cg, flag_cmg)
% Function to solve the linear system Ax = b
% x = solve_framework(x, A, b, flag_pcg, tol_cg, maxit_cg)
%
% INPUT:
% x - variables from previous step
% A - given matrix
% b - given vector
% flag_pcg - flag for the conjugate gradient method (0 or 1)
% maxit_cg - max number of iterations for the inner CG iterations
% tol_cg - stoping criterion for the inner CG iterations
% flag_cmg - with CMG precondition for the depth refinement(0 or 1) 

% OUTPUT:
% x - output variables 
%
% Author: Songyou Peng

% if (nargin<7) || isempty(flag_cmg)
%     flag_cmg = 0;
% end

if(flag_pcg)
    if size(A,1) == size(A,2)
        if (flag_cmg)
            M1 = cmg_sdd(A);
        else
            M1 = [];
        end
        M2 = [];
        [x,~] = pcg(A, b,tol_cg,maxit_cg,M1,M2, x);
    else
        M1 = [];
        M2 = [];
        [x,~] = pcg(A'*A, A'*b,tol_cg,maxit_cg,M1,M2,x);
    end
else
    x = A \ b;
end
    
end