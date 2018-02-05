from bpa import *
from bpa_misc import *

from misc import is_codeword

import numpy as np


def flip(bit):
    return (bit + 1) % 2

def gallager_b_rule(recvbit, msgs, t):
    if not msgs: return recvbit
    nmismatch = sum(1 for v in msgs if v != recvbit)
    if nmismatch >= t:
        return flip(recvbit)
    else:
        return recvbit

def gallager_a_rule(recvbit, msgs):
    return gallager_b_rule(recvbit, msgs, len(msgs))

def init_cn_gallager(r, incn):
    init_cn_inbox(r, incn, lambda x: x)

def majority(bitlist):
    if sum(bitlist) > len(bitlist) / 2:
        return 1
    else:
        return 0

def decide_gallager(r, invn):
    rhat = r.copy()
    for i in range(len(r)):
        if len(invn[i]) % 2: # odd vn degree
            rhat[i] = majority(invn[i].values())
        else:
            rhat[i] = majority([r[i]] + list(invn[i].values()))
    return rhat

def main():
    H = [[1, 1, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 0, 1]]
    r = [1, 1, 1, 1, 1]

    Ha = np.array(H)
    ra = np.array(r)
    max_iter = 100

    incn, outcn, invn, outvn = bpa_init(H)

    init_cn_gallager(r, incn)

    pfuncn = lambda i, xs : sum(xs) % 2
    pfunvn = lambda i, xs : gallager_a_rule(r[i], xs)
    need_stop = lambda invn : is_codeword(decide_gallager(ra, invn), Ha)

    bpa_loop(incn, outcn, invn, outvn, pfuncn, pfunvn, max_iter, need_stop)
    print(decide_gallager(ra, invn))

if __name__ == '__main__':
    main()
