"""
BCJR algorithm

Bit-wise maximum aposteriori decoding on a trellis.
"""
import bpa
from bpa_misc import decide_from_llrs

import numpy as np
from numpy import inf
from collections import defaultdict
from functools import reduce


def channel_metric(rsym, tr_out, sigmasq):
    return np.dot(rsym, tr_out) / sigmasq

def num_out(trellis):
    return len(trellis[0][0][0]['out'])

# are_llrs_in indicates whether llrs_prior are llrs on input or output bits.
def channel_metrics(r, trellis, sigmasq, llrs_prior, are_llrs_in):
    i = 0
    prevs, nexts = trellis
    nout = num_out(prevs)
    nllrs = 1 if are_llrs_in else nout
    rsymbols = np.reshape(r, (-1, nout))
    gammas = {}
    for rsym in rsymbols:
        #print(rsym)
        gammas[i] = {}
        l = llrs_prior[nllrs * i : nllrs * (i + 1)]
        #print(nllrs * i, nllrs * (i + 1), l)
        spec = 'in' if are_llrs_in else 'out'
        for f, ts in nexts[len(nexts) - 1 - i].items():
            gammas[i][f] = {}
            for t, transition in ts.items():
                adjust = np.dot(l, transition[spec]) * 0.5
                ch_metric = channel_metric(rsym, transition['out'], sigmasq)
                gammas[i][f][t] = adjust + ch_metric
        i += 1
    return gammas

def flip(channel_mtrx):
    flipped = {}
    for i in channel_mtrx.keys():
        flipped[len(channel_mtrx) - 1 - i] = bpa.flipinit(channel_mtrx[i])
    return flipped

def trellis_run(gammas, trellis, init_inf, orig):
    cum_metrics = np.zeros((len(gammas) + 1, len(trellis[0]))) # (num symbols + first state) x num states

    if init_inf:
        for j in np.arange(1, len(trellis[0])):
            cum_metrics[0][j] = -inf
    else:
        #print(orig[-1])
        cum_metrics[0] = orig[-1]

    for i in np.arange(1, len(gammas) + 1):
        for j in range(len(trellis[0])):
            #print(i, j, gammas, trellis[i-1][j].keys())
            metrics_for_paths = [ cum_metrics[i-1][k] + gammas[i-1][k][j] for k in trellis[i-1][j].keys() ]
            cum_metrics[i][j] = maxfun(metrics_for_paths)

    return cum_metrics

def maxfun(vals):
    #return max(vals)
    return max_star(vals)

def max_star_2(a, b):
    maxval = max(a, b)
    correction_term = 0 if a == -np.inf or b == -np.inf else np.log(1 + np.exp(-1 * np.abs(a - b)))
    return maxval + correction_term

def max_star(vals):
    return reduce(max_star_2, vals)
    #sum_exps = np.sum(np.exp(vals))
    #print(vals, sum_exps)
    #return -inf if sum_exps == 0.0 else np.log(sum_exps)
    #return np.log(sum_exps)

def bit_llrs(alphas, betas, gammas, trellis, is_in):
    num_symbols = len(gammas)
    nout = 1 if is_in else num_out(trellis)
    num_llrs = num_symbols * nout
    llrs = np.zeros(num_llrs)

    for i in range(num_symbols):
        for j in range(nout):
            s = 'in' if is_in else 'out'
            tr_plus  = [ (f, t) for t, v in trellis[i].items() for f, spec in v.items() if spec[s][j] == +1 ]
            tr_minus = [ (f, t) for t, v in trellis[i].items() for f, spec in v.items() if spec[s][j] == -1 ]

            v_plus  = maxfun([ alphas[i][f] + gammas[i][f][t] + betas[i+1][t] for f, t in tr_plus  ])
            v_minus = maxfun([ alphas[i][f] + gammas[i][f][t] + betas[i+1][t] for f, t in tr_minus ])

            llrs[i * nout + j] = v_plus - v_minus

    return llrs

def decode(r, trellis, sigma):
    llrs_prior = np.zeros(len(r))
    return decode_siso(r, trellis, sigma, llrs_prior, True, False, True)

def decode_siso(r, trellis, sigma, llrs_prior, is_in, are_llrs_in, init_inf):
    sigmasq = sigma ** 2
    prevs, nexts = trellis
    gammas = channel_metrics(r, trellis, sigmasq, llrs_prior, are_llrs_in)
    #print("g0:", gammas[0])
    #print("g1:", gammas[1])
    #print("g2:", gammas[2])
    #print("g3:", gammas[3])
    #print("gmid:", gammas[len(gammas) // 2])
    alphas = trellis_run(gammas, prevs, True, None)
    #print("alpha05:", alphas[0:5])
    #print("alphamid:", alphas[len(alphas) // 2])
    betas = np.flipud(trellis_run(flip(gammas), nexts, init_inf, alphas))
    llrs = bit_llrs(alphas, betas, gammas, prevs, is_in)
    #print("beta05:", betas[0:5])
    #print("betamid:", betas[len(betas) // 2])
    #print("llr05:", llrs[0:5])
    #print("llrmid:", llrs[len(llrs) // 2])
    #sys.exit(-1)
    #print(alphas)
    #print(betas)
    return llrs

def build_trellis(H):
    h_h, h_w = H.shape
    cum = np.zeros(h_h)
    num_states = 2 ** h_h

    nxt = {}
    for i in range(h_w):
        nxtIdx = h_w - 1 - i
        nxt[nxtIdx] = {}
        h_col = np.flipud(H[:,i]) # for some reason, column [1 0] means 2 not 1 in Ryan&Lin

        state = 0
        bitreprs = de2bi(np.arange(num_states), h_h)
        for bitrepr in bitreprs:
            nxt[nxtIdx][state] = {}
            nxt[nxtIdx][state][state] = {'in' : [+1], 'out' : [+1]}

            nextbitrepr = np.mod(bitrepr + h_col, 2)
            nextnumrepr = bi2de(nextbitrepr, h_h)

            nxt[nxtIdx][state][nextnumrepr] = {'in' : [-1], 'out' : [-1]}

            state += 1

    prev = flip(nxt)
    return prev, nxt

def de2bi(arnums, numbits):
    power = 2 ** np.arange(numbits)
    return np.floor((arnums[:,None]%(2*power))/power)

def bi2de(arbits, numbits):
    power = 2 ** np.arange(numbits)
    return int(np.sum(arbits * power))

def example_trellis():
    prev = {0 : { 0 : {'in' : [+1], 'out' : [+1, +1]}, 1 : {'in' : [+1], 'out' : [-1, -1]}},
            1 : { 2 : {'in' : [+1], 'out' : [-1, +1]}, 3 : {'in' : [+1], 'out' : [+1, -1]}},
            2 : { 0 : {'in' : [-1], 'out' : [-1, -1]}, 1 : {'in' : [-1], 'out' : [+1, +1]}},
            3 : { 2 : {'in' : [-1], 'out' : [+1, -1]}, 3 : {'in' : [-1], 'out' : [-1, +1]}}}

    return regular_trellis(prev)

def regular_trellis(prev):
    nxt = bpa.flipinit(prev)
    prevs = defaultdict(lambda: prev)
    nexts = defaultdict(lambda: nxt)
    return prevs, nexts

def test_conv_trellis():
    sigmasq = 1.0
    prevs, nexts = trellis = example_trellis()
    #r = [-0.7, -0.5, -0.8, -0.6, -1.1, 0.4, 0.9, 0.8, 0.0, -1.0]
    r = [0.0,  -0.5, 0.0,  -0.6, 0.0,  0.4, 0.0, 0.8, 0.0, -1.0]

    llrs_prior = np.zeros(len(r))
    gammas = channel_metrics(r, trellis, sigmasq, llrs_prior, True)
    alphas = trellis_run(gammas, prevs, True, None)
    betas = trellis_run(flip(gammas), nexts, True, None)
    betas_f = np.flipud(betas)

    #llrs = decode(r, trellis, 1.0)
    #rhat = decide_from_llrs(llrs)

    for i, k in gammas.items():
        print("Gammas ", i, k)

    for i, k in flip(gammas).items():
        print("Gammas flipped: ", i, k)

    print(alphas)
    print(betas)
    print(betas_f)

    llrs = bit_llrs(alphas, betas_f, gammas, prevs, False)
    rhat = decide_from_llrs(llrs)

    #for k, v in prev.items():
    #    print("PREV ", k, v)
    #for k, v in nxt.items():
    #    print("NEXT ", k, v)
    #print(r)

    print(bit_llrs(alphas, betas_f, gammas, prevs, True))
    print(decide_from_llrs(bit_llrs(alphas, betas_f, gammas, prevs, True)))
    print(llrs)
    print(rhat)

def test_block_trellis():
    H = [[1, 1, 0, 1, 0],
         [0, 1, 1, 0, 1]]
    H = np.array(H)
    prevs, nexts = trellis = build_trellis(H)

    print("NEXTs")
    for k, v in nexts.items():
        for f, t in v.items():
            print("SYM:", k, "FROM:", f, "TO:", t)

    print("PREVs")
    for k, v in prevs.items():
        for t, f in v.items():
            print("SYM:", k, "TO:", t, "FROM:", f)

    r = [1.2, 0.6, -1.2, 0.0, -0.1]
    llrs = decode(r, trellis, 1.0)
    rhat = decide_from_llrs(llrs)
    print(llrs)
    print(rhat)

def main():
    test_conv_trellis()
    #test_block_trellis()

if __name__ == '__main__':
    main()
