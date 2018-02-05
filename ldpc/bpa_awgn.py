from bpa import *
from bpa_misc import *
from misc import is_codeword

import numpy as np

def llr_awgn(y, sigmasq):
    return 2.0 * y / sigmasq

def init_cn_bpa_awgn(r, incn, sigmasq):
    init_cn_inbox(r, incn, lambda ri : llr_awgn(ri, sigmasq))

def cn_update_full_bpa(llrs):
    llrsa = np.array(llrs)
    return 2 * np.arctanh( np.prod( np.tanh( 0.5 * llrsa )))

def decide_from_llrs(llrs):
    return np.array([1 if llr < 0 else 0 for llr in llrs])

def total_bpa_llr(llrs_ch, invn):
    out_llrs = llrs_ch.copy()
    for i in range(len(llrs_ch)):
        out_llrs[i] = llrs_ch[i] + sum(invn[i].values())
    return out_llrs

def decide_bpa_llr(llrs_ch, invn):
    ret = decide_from_llrs(total_bpa_llr(llrs_ch, invn))
    return ret

def main():
    H = [[1, 0, 1, 0, 1, 0, 1],
         [0, 1, 1, 0, 0, 1, 1],
         [0, 0, 0, 1, 1, 1, 1]]
    r =  [-1, -1, -1, 1, 1, 1, 1]
    sigmasq = 1

    Ha = np.array(H)
    ra = np.array(r)
    max_iter = 100

    llrs_ch = [ llr_awgn(y, sigmasq) for y in r ]

    incn, outcn, invn, outvn = bpa_init(H)

    init_cn_bpa_awgn(r, incn, sigmasq)

    pfuncn = lambda i, xs : cn_update_full_bpa(xs)
    pfunvn = lambda i, xs : llrs_ch[i] + sum(xs)
    need_stop = lambda invn : is_codeword(decide_bpa_llr(llrs_ch, invn), Ha)

    bpa_loop(incn, outcn, invn, outvn, pfuncn, pfunvn, max_iter, need_stop)
    print(decide_bpa_llr(llrs_ch, invn))
    print(is_codeword(ra, Ha))


if __name__ == '__main__':
    main()
