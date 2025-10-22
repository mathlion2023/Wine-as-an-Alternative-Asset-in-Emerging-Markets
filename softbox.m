function p = softbox(x, lb, ub)
% Map unconstrained x to [lb,ub] via logistic
    p = lb + (ub - lb) ./ (1 + exp(-x));
end

function x = invsoftbox(p, lb, ub)
% Inverse mapping to unconstrained space
    z = (p - lb) ./ (ub - lb);
    z = min(max(z, 1e-10), 1-1e-10);
    x = -log(1./z - 1);
end
