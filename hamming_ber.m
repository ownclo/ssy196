[H, S_table] = hamming_15_11_tables();

% the plot was generated with maxRuns = 1e6
maxRuns = 1e3;
EbN0sdB = -2:10;
EbN0s = db2pow(EbN0sdB); % db

BERs_coded = arrayfun(@(ebn0) simulate_hamming_bsc(ebn0, H, S_table, maxRuns), EbN0s);
%BERs_coded_ref = bercoding(EbN0sdB,'Hamming','hard', 15);
BERs_uncoded = qfunc(sqrt(2 * EbN0s));
refBER = berawgn(EbN0sdB,'pam', 2);
figure(1);
semilogy(EbN0sdB, BERs_uncoded, '-*', 'LineWidth', 2); hold on;
semilogy(EbN0sdB, refBER, 'LineWidth', 2); hold on;
semilogy(EbN0sdB(1:end-1), BERs_coded(1:end-1), '-*', 'LineWidth', 2); hold on;
% Channel capacity came from non-included file
semilogy(-2:0.1:1.5, pbsR2(1:end-5), 'LineWidth', 2); hold on;
grid
legend('Uncoded BER','Reference Uncoded BER','Coded BER', 'C')
xlabel('Eb/No (dB)')
ylabel('BER')
hold off;


function ber = simulate_hamming_bsc(EbN0, H, S_table, maxRuns)

G = gen2par(H);

n = size(H, 2); % m = 4;
ncbits = gfrank(H); % n-k=4 - number of linearly-independent checks
k = n - ncbits; % k=11

p = qfunc(sqrt(2 * EbN0 * k / n));

ber = simulate_transmission(k, maxRuns,...
    @(w) encode_linear_code(G, w),...
    @(c) bsc_transmit(c, p),...
    @(r) syndrome_decode_hamming(H, S_table, r));
end
