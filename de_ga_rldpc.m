
ivls1 = de_ga_reg(3, 6, 1.162);
[ivls2, m1, m2] = de_ga_reg(3, 6, 1.1633);
ivls3 = de_ga_reg(3, 6, 1.165);
ivls4 = de_ga_reg(3, 6, 1.17);

figure(101)
plot(ivls1(1:find(ivls1, 1, 'last'))); hold on;
plot(ivls2(1:find(ivls2, 1, 'last'))); hold on;
plot(ivls3(1:find(ivls3, 1, 'last'))); hold on;
plot(ivls4(1:find(ivls4, 1, 'last'))); hold on;
ylim([0 10])
legend('1.162 dB', '1.163 dB', '1.165 dB', '1.17 dB')
xlabel('iteration');
ylabel('\mu(c)');
hold off;

figure(201)
m1 = m1(:,1:find(m1(1,:), 1, 'last'));
m2 = m2(:,1:find(m2(1,:), 1, 'last'));
plot(m1(2,:), m1(1,:), '-.'); hold on;
plot(m2(1,:), m2(2,:), '-.'); hold on;
ya = [m1(1,:) ; m2(2,:)];
xa = [m1(2,:) ; m2(1,:)];
plot([0 ; xa(:)], [0 ; ya(:)], '-'); hold on;
legend('C prev from V', 'C next from V');
xlabel('\mu^v');
ylabel('\mu^c');
hold off;

function [itervals, mu_v_from_mu, mu_c_from_mu_v] = de_ga_reg(dv, dc, ebno)
maxval = 10;
rate = 1 - dv / dc;

sigma = 1 / sqrt(2 * db2pow(ebno) * rate);
mu0 = 2 / sigma^2;
mu = 0;

itervals = zeros(1, 1000);
mu_v_from_mu = zeros(2, 1000);
mu_c_from_mu_v = zeros(2, 1000);
for i = 1:1000
    mu_v = mu0 + (dv - 1)*mu;
    mu_v_from_mu(:,i) = [mu mu_v]';
    mu = finv(@(m) F_approx(m), 1 - (1 - F_approx(mu_v))^(dc-1), 0, 100);
    mu_c_from_mu_v(:,i) = [mu_v mu]';
    itervals(i) = mu;
    if (mu > maxval)
        break;
    end
end
end

function res = F_approx(mu)
if (mu <= 10)
    res = exp(-0.4527 * mu^0.86 + 0.0218);
else
    res = sqrt(pi/mu) * exp(-mu/4) * (1 - 10/(7 * mu));
end
end

