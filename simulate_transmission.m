function ber = simulate_transmission(k, maxRuns, encodeFn, transmitFn, decodeFn)

totalErrors = 0;
totalBits = 0;

for i = 1:maxRuns
    word = randword(k);
    c = encodeFn(word);
    r = transmitFn(c);
    [~, word_hat] = decodeFn(r);
    numErrors = biterr(word, word_hat);
    totalErrors = totalErrors + numErrors;
    totalBits = totalBits + k;
end

ber = totalErrors / totalBits;
end


function u = randword(k)
%u = de2bi(randi([0 2^k-1]), k);
u = bsc_transmit(zeros(1, k), 0.5);
end
