%J(0.1)
%J_inv(0.0018)
%plot((1e-5:0.001:1.0), arrayfun(@(x) J(x), (1e-5:0.001:1.0)))

dv = 3; dc = 6;
ebn0 = 1.1; % dB
R = 1 - dv / dc;
sigmasq_ch = 8 * R * db2pow(ebn0);

iavs = 0.0:0.01:1.0;
ievs = arrayfun(@(iav) I_ev(dv, sigmasq_ch, iav), iavs);
iecs = arrayfun(@(iac) I_ec(dc, iac), iavs);

figure(200)
plot(iavs, ievs); hold on;
plot(iecs, iavs); hold on;
ylim([0 1]);
xlim([0 1]);
hold off;

function iev = I_ev(dv, sigmasq_ch, iav)
sigma_exit = sqrt( (dv - 1) * (J_inv(iav))^2 + sigmasq_ch );
iev = J(sigma_exit);
end

function iec = I_ec(dc, iac)
sigma_exit = sqrt( (dc - 1) * J_inv(1 - iac)^2 );
iec = 1 - J(sigma_exit);
end

function r = J(sigma)
sigmasq = sigma^2;
mu = sigmasq / 2;
if (sigma < 0.02)
    r = 0;
else
    r = 1 - integral(@(l) normpdf(l, mu, sigma) .* log2(1 + exp(-l)), -100, 100);
end
end

function sigma = J_inv(r)
sigma = finv(@(s) J(s), r, 1e-6, 20.0);
end