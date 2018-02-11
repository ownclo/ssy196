"""
BCJR algorithm

Bit-wise maximum aposteriori decoding on a trellis.
"""
import bpa
from bpa_awgn import decide_from_llrs

import numpy as np
from numpy import inf


def channel_metric(rsym, tr_out, sigmasq):
    return np.dot(rsym, tr_out) / sigmasq

def channel_metrics(r, trellis, sigmasq):
    i = 0
    prev, nxt = trellis
    rsymbols = np.reshape(r, (-1, len(prev[0][0]['out'])))
    gammas = {}
    for rsym in rsymbols:
        gammas[i] = {}
        for f, ts in nxt.items():
            gammas[i][f] = {}
            for t, transition in ts.items():
                gammas[i][f][t] = channel_metric(rsym, transition['out'], sigmasq)
        i += 1
    return gammas

def flip(channel_mtrx):
    flipped = {}
    for i in channel_mtrx.keys():
        flipped[len(channel_mtrx) - 1 - i] = bpa.flipinit(channel_mtrx[i])
    return flipped

def trellis_run(gammas, trellis):
    cum_metrics = np.zeros((len(gammas) + 1, len(trellis))) # (num symbols + first state) x num states

    for j in np.arange(1, len(trellis)):
        cum_metrics[0][j] = -inf

    for i in np.arange(1, len(gammas) + 1):
        for j in range(len(trellis)):
            metrics_for_paths = [ cum_metrics[i-1][k] + gammas[i-1][k][j] for k in trellis[j].keys() ]
            cum_metrics[i][j] = maxfun(metrics_for_paths)

    return cum_metrics

def maxfun(vals):
    #return max(vals)
    return max_star(vals)

def max_star(vals):
    return np.log(np.sum(np.exp(vals)))

def bit_llrs(alphas, betas, gammas, trellis):
    num_symbols = len(gammas)
    llrs = np.zeros(num_symbols)
    tr_plus = [ (f, t) for t, v in trellis.items() for f, spec in v.items() if spec['in'] == [+1] ]
    tr_minus = [ (f, t) for t, v in trellis.items() for f, spec in v.items() if spec['in'] == [-1] ]

    for i in np.arange(num_symbols):
        v_plus  = maxfun([ alphas[i][f] + gammas[i][f][t] + betas[i+1][t] for f, t in tr_plus  ])
        v_minus = maxfun([ alphas[i][f] + gammas[i][f][t] + betas[i+1][t] for f, t in tr_minus ])
        llrs[i] = v_plus - v_minus

    return llrs

def decode(r, trellis):
    sigmasq = 1.0
    prev, nxt = trellis
    gammas = channel_metrics(r, trellis, sigmasq)
    alphas = trellis_run(gammas, prev)
    betas = np.flipud(trellis_run(flip(gammas), nxt))
    llrs = bit_llrs(alphas, betas, gammas, prev)
    return llrs

def build_trellis():
    prev = {0 : { 0 : {'in' : [+1], 'out' : [+1, +1]}, 1 : {'in' : [+1], 'out' : [-1, -1]}},
            1 : { 2 : {'in' : [+1], 'out' : [-1, +1]}, 3 : {'in' : [+1], 'out' : [+1, -1]}},
            2 : { 0 : {'in' : [-1], 'out' : [-1, -1]}, 1 : {'in' : [-1], 'out' : [+1, +1]}},
            3 : { 2 : {'in' : [-1], 'out' : [+1, -1]}, 3 : {'in' : [-1], 'out' : [-1, +1]}}}

    nxt = bpa.flipinit(prev)
    return prev, nxt

def main():
    sigmasq = 1.0
    prev, nxt = trellis = build_trellis()
    r = [-0.7, -0.5, -0.8, -0.6, -1.1, 0.4, 0.9, 0.8, 0.0, -1.0]

    gammas = channel_metrics(r, trellis, sigmasq)
    alphas = trellis_run(gammas, prev)
    betas = trellis_run(flip(gammas), nxt)
    betas_f = np.flipud(betas)

    for i, k in gammas.items():
        print("Gammas ", i, k)

    for i, k in flip(gammas).items():
        print("Gammas flipped: ", i, k)

    print(alphas)
    print(betas)
    print(betas_f)

    llrs = bit_llrs(alphas, betas_f, gammas, prev)
    rhat = decide_from_llrs(llrs)

    #for k, v in prev.items():
    #    print("PREV ", k, v)
    #for k, v in nxt.items():
    #    print("NEXT ", k, v)
    #print(r)

    print(llrs)
    print(rhat)

if __name__ == '__main__':
    main()
