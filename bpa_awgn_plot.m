load('./ldpc/bpa_awgn_4.mat')
load('./ldpc/ml_awgn_4.mat')
load('./ldpc/bpa_awgn_min_2.mat')
load('./ldpc/uncoded.mat')
load('./ldpc/bpa_bec_2.mat')
load('./ldpc/bpa_awgn_h21.mat')
load('./ldpc/bpa_awgn_bcjr.mat')
load('./ldpc/bpa_awgn_bcjr_2.mat')
load('./ldpc/pccc_7_5.mat')
load('./ldpc/pccc_7_5_punc.mat')
load('./ldpc/pccc_7_5_punc_ref.mat')
load('./ldpc/sccc_7_5.mat')
%load('./ldpc/sccc_7_1.mat')
load('./ldpc/sccc_7_5_tst.mat') % ACTUALLY 7,1!
%load('./ldpc/pccc_unpunctured_7_5.mat')
load('./ldpc/sccc_7_5_fancypunc.mat')

%2.4 8e-07 0.0003
sb = [0.5 1 1.5 2 2.4 2.5];
bers = [0.198666 0.12775 0.030666 0.000238276 8e-07 2.9850746e-07];
fers = [1.0 0.75 0.46666666 0.0088719 0.0003 7.462e-05];

EbN0sdB = -2:10;
hd_bers = bercoding(EbN0sdB, 'Hamming','hard', 7);

figure(300)
semilogy(bpa_awgn_snrs_4, bpa_awgn_bers_4, '-.', 'LineWidth', 2); hold on;
semilogy(ml_awgn_snrs_4, ml_awgn_bers_4, '--o', 'LineWidth', 2); hold on;
%%semilogy(bpa_awgn_min_snrs_2, bpa_awgn_min_bers_2, '--*', 'LineWidth', 2); hold on;
semilogy(bpa_bec_snrs_2, bpa_bec_bers_2, '-*', 'LineWidth', 2); hold on;
semilogy(uncsnrs, uncbers, '-', 'LineWidth', 2); hold on;
semilogy(EbN0sdB, hd_bers, '--.', 'LineWidth', 2); hold on;
semilogy(bpa_awgn_snrs_h21, bpa_awgn_bers_h21, '-.', 'LineWidth', 2); hold on;
%%semilogy(bpa_awgn_snrs_bcjr, bpa_awgn_bers_bcjr, '--*', 'LineWidth', 2); hold on;
semilogy(bpa_awgn_snrs_bcjr_2, bpa_awgn_bers_bcjr_2, '--*', 'LineWidth', 2); hold on;
%semilogy(pccc_7_5_snrs, pccc_7_5_bers, '-*', 'LineWidth', 2); hold on;
semilogy(pccc_7_5_snrs_punc_ref, pccc_7_5_bers_punc_ref, '--*', 'LineWidth', 2); hold on;
semilogy(pccc_7_5_snrs_punc_ref, pccc_7_5_fers_punc_ref, '--*', 'LineWidth', 2); hold on;
%semilogy(pccc_7_5_snrs_punc, pccc_7_5_bers_punc, '--*', 'LineWidth', 2); hold on;
%semilogy(sccc_7_5_snrs, sccc_7_5_bers, '-o', 'LineWidth', 2); hold on;
%semilogy(sccc_7_5_snrs, sccc_7_5_fers, '-o', 'LineWidth', 2); hold on;
%semilogy(pccc_unpunctured_7_5_snrs, pccc_unpunctured_7_5_bers, '-*', 'LineWidth', 2); hold on;
semilogy(sb, bers, '-o', 'LineWidth', 2); hold on;
semilogy(sb, fers, '-o', 'LineWidth', 2); hold on;
%semilogy(sccc_7_1_snrs, sccc_7_1_bers, '-o', 'LineWidth', 2); hold on;
%semilogy(sccc_7_1_snrs, sccc_7_1_fers, '-o', 'LineWidth', 2); hold on;
semilogy(sccc_7_5_tst_snrs, sccc_7_5_tst_bers, '-o', 'LineWidth', 2); hold on; %% 7 1 !!!
semilogy(sccc_7_5_tst_snrs, sccc_7_5_tst_fers, '-o', 'LineWidth', 2); hold on; %% 7 1 !!!
semilogy(sccc_7_5_fancypunc_snrs, sccc_7_5_fancypunc_bers, '-o', 'LineWidth', 2); hold on;
semilogy(sccc_7_5_fancypunc_snrs, sccc_7_5_fancypunc_fers, '-o', 'LineWidth', 2); hold on;

xlim([-2, 8]);
%xlim([0, 5]);
grid minor

%legend('BPA','ML','BPA-MIN', 'BPA-BEC', 'uncoded', 'hard-decision')
legend('BPA','ML', 'BPA-BEC', 'uncoded', 'hard-decision', 'H_{21}', 'BCJR', 'PCCC BER', 'PCCC FER', 'SCCC BER', 'SCCC FER')
%legend('uncoded', 'PCCC BER', 'PCCC FER', 'SCCCC BER', 'SCCC FER')
xlabel('Eb/No (dB)')
ylabel('P_e')

hold off;