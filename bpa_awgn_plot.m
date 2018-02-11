load('./ldpc/bpa_awgn_4.mat')
load('./ldpc/ml_awgn_3.mat')
load('./ldpc/bpa_awgn_min_2.mat')
load('./ldpc/uncoded.mat')
load('./ldpc/bpa_bec_2.mat')
load('./ldpc/bpa_awgn_h21.mat')

EbN0sdB = -2:10;
hd_bers = bercoding(EbN0sdB, 'Hamming','hard', 7);

figure(1)
semilogy(bpa_awgn_snrs_4, bpa_awgn_bers_4, '-.', 'LineWidth', 2); hold on;
semilogy(ml_awgn_snrs_3, ml_awgn_bers_3, '--o', 'LineWidth', 2); hold on;
%semilogy(bpa_awgn_min_snrs_2, bpa_awgn_min_bers_2, '--*', 'LineWidth', 2); hold on;
semilogy(bpa_bec_snrs_2, bpa_bec_bers_2, '-*', 'LineWidth', 2); hold on;
semilogy(uncsnrs, uncbers, '-', 'LineWidth', 2); hold on;
semilogy(EbN0sdB, hd_bers, '--.', 'LineWidth', 2); hold on;
semilogy(bpa_awgn_snrs_h21, bpa_awgn_bers_h21, '-.', 'LineWidth', 2); hold on;
xlim([-2, 8]);
grid minor

%legend('BPA','ML','BPA-MIN', 'BPA-BEC', 'uncoded', 'hard-decision')
legend('BPA','ML', 'BPA-BEC', 'uncoded', 'hard-decision', 'H_{21}')
xlabel('Eb/No (dB)')
ylabel('BER')

hold off;