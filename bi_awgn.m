R1 = 1/2;
R2 = 11/15;

%step = 0.01;
%ebn0sdBR1 = -0.4 : step : 0.2;
%ebn0sdBR2 = -
%ebn0s = db2pow(ebn0sdB);
%pbsR1 = pb_constrained_snrs(ebn0s, R1);
pbsR2 = pb_constrained_snrs(db2pow(-2:0.1:2), R2);
pbs = logspace(-1.9, -5);
ebn0s1 = snr_constrained_pb(pbs, R1);
ebn0s2 = snr_constrained_pb(pbs, R2);
figure(1);
%semilogy(ebn0sdB, pbsR1); hold on;
%semilogy(ebn0sdB, pbsR2); hold on;
%semilogy(pow2db(ebn0s1), pbs); hold on;
semilogy(pow2db(ebn0s2), pbs); hold on;
%xlim([-0.4 -0.2]);
grid on;
hold off;
%inv_capacity_bi_awgn(0.9999)

function pbs = pb_constrained_snrs(ebn0s, R)

sigmas = 1 ./ sqrt(2 * R * ebn0s);
Rc = arrayfun(@(sigma) capacity_bi_awgn(sigma), sigmas);
hpbs = 1 - Rc / R;

pbs = arrayfun(@(hpb) hbinv(hpb), hpbs);
end

function ebn0s = snr_constrained_pb(pbs, R)
Rc = R * (1 - hbin(pbs));
sigmas = arrayfun(@(c) inv_capacity_bi_awgn(c), Rc);
ebn0s = (1/(2*R)) .* sigmas.^(-2);
end

function C = capacity_bi_awgn(sigma)
%y = gen_bi_awgn_output(sigma, len);
%p_y = prob_bi_awgn_output(y, sigma);
%H_y = -1 * mean(log2(p_y));
H_y = integral(@(y) infm(prob_bi_awgn_output(y, sigma)), -1-5*sigma, 1+5*sigma);
C = H_y - 0.5*log2(2 * pi * exp(1) * sigma^2);
end

function pr = hbinv(entropy)
pr = fminbnd(@(k) abs(hbin(k) - entropy), 0, 0.5);
end

function sigma = inv_capacity_bi_awgn(C)
sigma = fminbnd(@(sigma) abs(capacity_bi_awgn(sigma) - C), 0.01, 10);
end

function h = hbin(p)
h = infm(p) + infm(1-p);
end

function bs = infm(p)
bs = -p .* log2(p);
end

%function [y] = gen_bi_awgn_output(sigma, len)
%x = (-1).^randi([0 1], 1, len);
%y = x + sigma .* randn(1, len);
%end

function [p] = prob_bi_awgn_output(y, sigma)
p = 0.5 * (normpdf(y, 1, sigma) + normpdf(y, -1, sigma));
end