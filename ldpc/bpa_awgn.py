from bpa import *
from bpa_misc import *
from misc import is_codeword, db2pow, last_k
from simulate_transmission import simulate_transmission
import bawgn
import maximum_likelihood as ml

import numpy as np
import scipy.io
from scipy.stats import norm
from random import random


def llr_awgn(y, sigmasq):
    return 2.0 * y / sigmasq

def llr_bec(y, sigmasq):
    #threshold = np.sqrt(sigmasq)
    threshold = 0.4
    if np.abs(y) > threshold:
        return np.sign(y) * llr_bit(threshold, sigmasq)
    else:
        return 0

def llr_bit(t, sigmasq):
    sigma = np.sqrt(sigmasq)
    pmatch = norm.cdf(1.0 - t, sigma)
    pmismatch = norm.cdf(-1.0 - t, sigma)
    llr = np.log(pmatch / pmismatch)
    return llr

def init_cn_bpa_awgn(r, incn, sigmasq):
    init_cn_inbox(r, incn, lambda ri : llr_awgn(ri, sigmasq))

def init_cn_bpa_bec(r, incn, sigmasq):
    init_cn_inbox(r, incn, lambda ri : llr_bec(ri, sigmasq))


def cn_update_full_bpa(llrs):
    llrsa = np.array(llrs)
    return 2 * np.arctanh( np.prod( np.tanh( 0.5 * llrsa )))

def cn_update_min(llrs):
    sign = np.prod(np.sign(llrs))
    val = np.min(np.abs(llrs))
    return sign * val

def decide_from_llrs(llrs):
    return np.array([decide_from_llr(llr) for llr in llrs])

def decide_from_llr(llr):
    if llr == 0: return 0 if random() > 0 else 1
    else: return 1 if llr < 0 else 0

def total_bpa_llr(llrs_ch, invn):
    out_llrs = llrs_ch.copy()
    for i in range(len(llrs_ch)):
        out_llrs[i] = llrs_ch[i] + sum(invn[i].values())
    return out_llrs


def decide_bpa_llr(llrs_ch, invn):
    return decide_from_llrs(total_bpa_llr(llrs_ch, invn))


# BPA decide stage demodulates automatically
def decode_bpa_awgn(msg_store, r, H, sigma, max_iter):
    incn, outcn, invn, outvn = msg_store
    sigmasq = sigma ** 2
    llrs_ch = [ llr_awgn(y, sigmasq) for y in r ]

    init_cn_bpa_awgn(r, incn, sigmasq)
    return decode_bpa(msg_store, llrs_ch, max_iter, H)


def decode_bpa_bec(msg_store, r, H, sigma, max_iter):
    incn, outcn, invn, outvn = msg_store
    sigmasq = sigma ** 2
    llrs_ch = [ llr_bec(y, sigmasq) for y in r ]

    init_cn_bpa_bec(r, incn, sigmasq)
    return decode_bpa(msg_store, llrs_ch, max_iter, H)


def decode_bpa(msg_store, llrs_ch, max_iter, H):
    incn, outcn, invn, outvn = msg_store

    #pfuncn = lambda i, xs : cn_update_full_bpa(xs)
    pfuncn = lambda i, xs : cn_update_min(xs)
    pfunvn = lambda i, xs : llrs_ch[i] + sum(xs)
    need_stop = lambda invn : is_codeword(decide_bpa_llr(llrs_ch, invn), H)
    #need_stop = lambda invn : False

    bpa_loop(incn, outcn, invn, outvn, pfuncn, pfunvn, max_iter, need_stop)
    c_hat = decide_bpa_llr(llrs_ch, invn)
    return c_hat


# simulate BPA for Hamming code over BI-AWGN
def simulate_bpa_awgn():
    H = [[1, 0, 0, 1, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 1],
         [0, 0, 1, 0, 1, 1, 1]]
    n = 7
    k = 4
    Ha = np.array(H)

    # max number of iterations of iterative decoding session (per word!)
    max_iter = 100

    msg_store = bpa_init(H)

    # CW book
    cws = ml.all_codewords()

    # max runs of simulations (number of words to be transmitted)
    max_runs = 100000
    snrDbs = np.arange(-2, 10)
    bers = np.zeros(len(snrDbs))
    i = 0
    for snrDb in snrDbs:
        snr = db2pow(snrDb)
        sigma = bawgn.noise_sigma(snr, k / n)

        #decode = lambda r : decode_bpa_awgn(msg_store, r, Ha, sigma, max_iter)
        #decode = lambda r : ml.decode_ml(cws, r)
        decode = lambda r : decode_bpa_bec(msg_store, r, Ha, sigma, max_iter)
        ber = simulate_transmission(n, k, max_runs, bawgn.modulate, bawgn.transmit(sigma), decode)

        bers[i] = ber
        i += 1
        print(snrDb, ber)
    # save to mat file for plotting purposes
    #scipy.io.savemat('ml_awgn_2', { 'ml_awgn_snrs_2' : snrDbs, 'ml_awgn_bers_2' : bers })
    #scipy.io.savemat('bpa_awgn_min_2', { 'bpa_awgn_min_snrs_2' : snrDbs, 'bpa_awgn_min_bers_2' : bers })
    scipy.io.savemat('bpa_bec', { 'bpa_bec_snrs' : snrDbs, 'bpa_bec_bers' : bers })


def test_main():
    H = [[1, 0, 1, 0, 1, 0, 1],
         [0, 1, 1, 0, 0, 1, 1],
         [0, 0, 0, 1, 1, 1, 1]]
    r =  [1, -1, -1, 1, 1, 1, 1]
    sigma = 1
    max_iter = 100

    Ha = np.array(H)
    ra = np.array(r)

    # can be done once
    msg_store = bpa_init(H)

    c_hat = decode_bpa_awgn(msg_store, ra, Ha, sigma, max_iter)

    print(c_hat)
    print(is_codeword(bawgn.demodulate_hard(ra), Ha))
    print(is_codeword(c_hat, Ha))


def main():
    simulate_bpa_awgn()


if __name__ == '__main__':
    main()
