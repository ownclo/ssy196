from bpa import *
from bpa_misc import *
from misc import is_codeword, db2pow, last_k
from simulate_transmission import simulate_transmission
import bawgn
import maximum_likelihood as ml
import bcjr

import numpy as np
import scipy.io
from scipy.stats import norm
from random import random


def llr_awgn(y, sigmasq):
    return 2.0 * y / sigmasq

def llr_bec(y, sigmasq):
    #threshold = np.sqrt(sigmasq)
    threshold = 0.2
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

def vn_update_full_bpa(llr_ch, llrs_inc):
    #print("LLR CH: ", llr_ch, "LLRS_incoming: ", llrs_inc)
    llr = llr_ch + sum(llrs_inc)
    #if len(llrs_inc) == 0: print(llr_ch, llr)
    return llr

def cn_update_min(llrs):
    sign = np.prod(np.sign(llrs))
    val = np.min(np.abs(llrs))
    return sign * val

def total_bpa_llr(llrs_ch, invn):
    out_llrs = llrs_ch.copy()
    for i in range(len(llrs_ch)):
        out_llrs[i] = llrs_ch[i] + sum(invn[i].values())
    #print("ITER LLR: ", out_llrs)
    return out_llrs


def decide_bpa_llr(llrs_ch, invn):
    return decide_from_llrs(total_bpa_llr(llrs_ch, invn))


# BPA decide stage demodulates automatically
def decode_bpa_awgn(msg_store, r, H, sigma, max_iter):
    incn, outcn, invn, outvn = msg_store
    sigmasq = sigma ** 2
    llrs_ch = [ llr_awgn(y, sigmasq) for y in r ]
    #print("LLRS ch: ", llrs_ch)

    init_cn_bpa_awgn(r, incn, sigmasq)
    #print(incn)
    d_bpa = decode_bpa(msg_store, llrs_ch, max_iter, H)
    #d_ml = ml.decode_ml(cws, r)
    #if np.any(d_bpa - d_ml): print(d_bpa, d_ml, r)
    return d_bpa


def decode_bpa_bec(msg_store, r, H, sigma, max_iter):
    incn, outcn, invn, outvn = msg_store
    sigmasq = sigma ** 2
    llrs_ch = [ llr_bec(y, sigmasq) for y in r ]

    init_cn_bpa_bec(r, incn, sigmasq)
    return decode_bpa(msg_store, llrs_ch, max_iter, H)


def decode_bpa(msg_store, llrs_ch, max_iter, H):
    incn, outcn, invn, outvn = msg_store

    pfuncn = lambda i, xs : cn_update_full_bpa(xs)
    #pfuncn = lambda i, xs : cn_update_min(xs)
    pfunvn = lambda i, xs : vn_update_full_bpa(llrs_ch[i], xs) 
    need_stop = lambda invn : is_codeword(decide_bpa_llr(llrs_ch, invn), H)
    #need_stop = lambda invn : False

    bpa_loop(incn, outcn, invn, outvn, pfuncn, pfunvn, max_iter, need_stop)
    c_hat = decide_bpa_llr(llrs_ch, invn)
    return c_hat


# simulate BPA for Hamming code over BI-AWGN
def simulate_bpa_awgn():
    #H = [[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    #     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    #     [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]]
    #n = 21
    #k = 12

    H = [[1, 0, 0, 1, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 1],
         [0, 0, 1, 0, 1, 1, 1]]
    n = 7
    k = 4
    Ha = np.array(H)
    n = Ha.shape[1]
    k = n - Ha.shape[0]

    # max number of iterations of iterative decoding session (per word!)
    max_iter = 10

    msg_store = bpa_init(H)
    #print(msg_store)

    # CW book
    cws = ml.all_codewords()

    # BCRJ trellis
    trellis = bcjr.build_trellis(Ha)

    # max runs of simulations (number of words to be transmitted)
    max_runs = 1e7
    snrDbs = np.arange(-2, 9)
    bers = np.zeros(len(snrDbs))
    i = 0
    for snrDb in snrDbs:
        snr = db2pow(snrDb)
        sigma = bawgn.noise_sigma(snr, k / n)

        #decode = lambda r : decode_bpa_awgn(msg_store, r, Ha, sigma, max_iter)
        #decode = lambda r : ml.decode_ml(cws, r)
        #decode = lambda r : decode_bpa_bec(msg_store, r, Ha, sigma, max_iter)
        decode = lambda r : decide_from_llrs(bcjr.decode(r, trellis, sigma))
        ber = simulate_transmission(n, n, max_runs, bawgn.modulate, bawgn.transmit(sigma), decode)

        bers[i] = ber
        i += 1
        print(snrDb, ber)
    # save to mat file for plotting purposes
    #scipy.io.savemat('ml_awgn_4', { 'ml_awgn_snrs_4' : snrDbs, 'ml_awgn_bers_4' : bers })
    #scipy.io.savemat('bpa_awgn_4', { 'bpa_awgn_snrs_4' : snrDbs, 'bpa_awgn_bers_4' : bers })
    #scipy.io.savemat('bpa_bec_2', { 'bpa_bec_snrs_2' : snrDbs, 'bpa_bec_bers_2' : bers })
    #scipy.io.savemat('bpa_awgn_h21', { 'bpa_awgn_snrs_h21' : snrDbs, 'bpa_awgn_bers_h21' : bers })
    scipy.io.savemat('bpa_awgn_bcjr_3', { 'bpa_awgn_snrs_bcjr_3' : snrDbs, 'bpa_awgn_bers_bcjr_3' : bers })


def test_main():
    H = [[1, 0, 1, 0, 1, 0, 1],
         [0, 1, 1, 0, 0, 1, 1],
         [0, 0, 0, 1, 1, 1, 1]]
    #[1 0 0 0 0 0 0] [0 0 0 0 0 0 0]
    r = [-0.34808075,  1.26110584,  0.49168568, 0.29461952,  1.1177641,   0.92850077, 1.25788416]

    #sigma = 1
    sigma = bawgn.noise_sigma(db2pow(8), 4 / 7)
    print(sigma)

    cws = ml.all_codewords()

    max_iter = 5

    Ha = np.array(H)
    ra = np.array(r)

    # can be done once
    msg_store = bpa_init(H)

    c_hat = decode_bpa_awgn(msg_store, ra, Ha, sigma, max_iter, cws)

    print(c_hat)
    print(is_codeword(bawgn.demodulate_hard(ra), Ha))
    print(is_codeword(c_hat, Ha))


def main():
    #test_main()
    simulate_bpa_awgn()


if __name__ == '__main__':
    main()
