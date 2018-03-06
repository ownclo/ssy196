import bawgn
from misc import db2pow

import numpy as np
from scipy.stats import norm
import scipy.io


def simulate_transmission(n, decoded_n, max_runs, encode, transmit, decode):
    max_errors = 500
    max_frames_in_error = 7
    total_errors = 0.0
    total_bits = 0.0
    frames_in_error = 0
    word = np.zeros(n) # will always transmit all-zero codeword

    i = 0
    #while (i < max_runs or total_errors == 0):
    for i in range(round(max_runs)):
        c = encode(word)
        r = transmit(c)
        word_hat = decode(r)
        num_errors = weight(word_hat)  # if we've transmitted all-zero codeword
        if num_errors > 0:
            frames_in_error += 1
            print(np.array([i for i in range(len(word_hat)) if word_hat[i] != 0]))
        total_errors += num_errors
        total_bits += decoded_n
        i += 1
        if total_errors >= max_errors:
            print("BER break")
            break
        if frames_in_error >= max_frames_in_error:
            print("FER break")
            break
        if i % 100 == 0:
            print("REPORT. i:", i, "frames_in_error:", frames_in_error, "total_errors:", total_errors, "FER:", frames_in_error / i, "BER:", total_errors / total_bits)

    ber = total_errors / total_bits
    fer = frames_in_error / i
    return ber, fer


def weight(word):
    return sum(word)


# simulate uncoded transmission
def main():
    max_runs = 100000
    snrDbs = np.arange(-2, 10)
    bers = np.zeros(len(snrDbs))
    i = 0
    for snrDb in snrDbs:
        snr = db2pow(snrDb)
        sigma = bawgn.noise_sigma(snr, 1) # uncoded transmission
        n = 100
        ber = simulate_transmission(n, n, max_runs, bawgn.modulate, bawgn.transmit(sigma), bawgn.demodulate_hard)
        bers[i] = ber
        i += 1
        th = norm.sf(1 / sigma)
        print(snrDb, ber, th)
    # save to mat file for plotting purposes
    scipy.io.savemat('uncoded', { 'uncsnrs' : snrDbs, 'uncbers' : bers })


if __name__ == '__main__':
    main()
