p = de_bec_reg(3, 6, 0.4294)
dv = 3;
dc = 6;
eps_star = binsearch(@(eps) de_bec_reg(dv, dc, eps) > 1e-16, 0.0, 1.0, 1e-6)
%l = fliplr([0 0.2343 0.3406 0 0 0 0.2967 0.1284]);
%r = fliplr([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.3 0.7]);
l = [0.1284 0.2967 0 0 0 0.3406 0.2343 0];
r = [0.7 0.3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
%l = [1 0 0];
%r = [1 0 0 0 0 0];
eps_star2 = binsearch(@(eps) de_bec_ireg(l, r, eps) > 1e-16, 0.0, 1.0, 1e-6)

function p = de_bec_reg(dv, dc, eps)

% NB! maxIterations will considerably affect your threshold.
% The larger maxIterations, the further to the right is the threshold
maxIterations = 10000;

% p - prob of erasure going from VN to CN
% q - prob of erasure going from CN to VN
p = eps;

for i = 1 : maxIterations
    q = 1 - (1 - p)^(dc-1);
    p = eps * q^(dv-1);
    if (p < 1e-16), break; end
end
end

function p = de_bec_ireg(l, r, eps)

% NB! maxIterations will considerably affect your threshold.
% The larger maxIterations, the further to the right is the threshold
maxIterations = 10000;
p = eps;

for i = 1 : maxIterations
    q = 1 - polyval(r, 1 - p);
    p = eps * polyval(l, q);
    if (p < 1e-16), break; end
end
end
