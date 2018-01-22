[H, S_table] = hamming_15_11_tables();

% plot was done with maxRuns = 1e4
maxRuns = 1e3;
EbN0sdB = -2:10;
EbN0s = db2pow(EbN0sdB); % db

BERs_1 = arrayfun(@(ebn0) simulate_product_hamming_bsc(ebn0, H, S_table, 1, maxRuns), EbN0s);
BERs_2 = arrayfun(@(ebn0) simulate_product_hamming_bsc(ebn0, H, S_table, 2, maxRuns), EbN0s);
BERs_3 = arrayfun(@(ebn0) simulate_product_hamming_bsc(ebn0, H, S_table, 3, maxRuns), EbN0s);


figure(2);
semilogy(EbN0sdB, BERs_1, '-*', 'LineWidth', 2); hold on;
semilogy(EbN0sdB(1:end-4), BERs_2(1:end-4), '-o', 'LineWidth', 2); hold on;
semilogy(EbN0sdB, BERs_3, '-+', 'LineWidth', 3); hold on;
grid
legend('1 iteration','2 iterations', '3 iterations')
xlabel('Eb/No (dB)')
ylabel('BER')
hold off;

% w = zeros(1, 11 * 11);
% w(1,3) = 1;
% w
% c = encode_product_code(gen2par(H), 15, 11, gen2par(H), 15, 11, w)
% [~, word_hat] = syndrome_decode_hamming_product(H, S_table, 11, H, S_table, 11, 1, c);
% word_hat
% isequal(w, word_hat)

function ber = simulate_product_hamming_bsc(ebn0, H, S_table, numIterations, maxRuns)
G = gen2par(H);

n = size(H, 2); % m = 4;
ncbits = gfrank(H); % n-k=4 - number of linearly-independent checks
k = n - ncbits; % k=11

n_product = n * n;
k_product = k * k;
r_product = k_product / n_product;

p = qfunc(sqrt(2 * ebn0 * r_product));

ber = simulate_transmission(k_product, maxRuns,...
    @(w) encode_product_code(G, n, k, G, n, k, w),...
    @(c) bsc_transmit(c, p),...
    @(r) syndrome_decode_hamming_product(H, S_table, k, H, S_table, k, numIterations, r));
end


% first stage - encode columns —
% second stage - encode rows |
% Basic assumption everywhere throughout this code:
% rows are encoded with code G1, columns are encoded with code G2
function c = encode_product_code(G1, n1, k1, G2, n2, k2, w)

w_mat = arrange_into_matrix(w, k1, k2);

c = zeros(n1, n2);
% for each row:
for i = 1:size(w_mat, 1)
    c(i,:) = encode_linear_code(G1, w_mat(i,:));
end

% for each column in c
for j = 1:size(c, 2)
    c(:,j) = encode_linear_code(G2, c(1:k2, j).').';
end
end


% there is k1 columns by our assumption above.
function w_mat = arrange_into_matrix(w, k1, ~)
w_mat = vec2mat(w, k1);
end

% convert matrix back into array row-by-row
function w = arrange_into_array(w_mat)
w = w_mat.';
w = w(:).';
end

function [c_hat, word_hat] = syndrome_decode_hamming_product(H1, S_table1, k1, H2, S_table2, k2, numIterations, r)
c_mat = r;

for iter = 1:numIterations
    % for each row - decode with code 1
    for i = 1:size(r, 1)
        c_mat(i,:) = syndrome_decode_hamming(H1, S_table1, c_mat(i,:));
    end

    % for each column - decode with code 2
    for j = 1:size(r, 2)
        c_mat(:,j) = syndrome_decode_hamming(H2, S_table2, c_mat(:,j).').';
    end
end
word_hat = arrange_into_array(c_mat(end-k2+1:end,end-k1+1:end));
c_hat = arrange_into_array(c_mat);
end
