from simulate_transmission import simulate_transmission
import bawgn
from bpa_awgn import decide_from_llrs, llr_awgn
from misc import db2pow
import bcjr

import numpy as np
from itertools import combinations
import scipy.io

np.set_printoptions(linewidth=150)


def delay(u, d):
    return np.roll(u, d)

def zeros(l):
    return np.zeros(l, dtype='int')

def empty(l):
    return np.empty(l, dtype='int')

def zero_pad(u, new_len):
    ret = zeros(new_len)
    ret[0:len(u)] = u
    return ret

def an_interleaver(k):
    p = zeros(k)
    p[0] = 11
    for i in np.arange(1, k):
        p[i] = (13 * p[i - 1] + 7) % k
    return p

def rand_interleaver(k):
    return np.random.permutation(k)

# buggy (prone to deadlock loop and emergency escape) version
def s_interleaver(k, s):
    used = set()
    whole = set(np.arange(k))
    perm = zeros(k)
    for i in range(k):
        possible = list(whole - used)
        iteration = 0
        while True:
            candidate = np.random.choice(possible)
            distances = np.array([abs(perm[max(0, i - j)] - candidate) for j in np.arange(1, s)])
            #print(possible, distances)
            iteration += 1
            if all(distances > s) or iteration > k: break
        used.add(candidate)
        perm[i] = candidate
    return perm

def deinterleaver(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s

def interleave(word, interleaver):
    return word[interleaver]

def encode_conv_7_5(u):
    start_state = zeros(2)
    return __encode_conv_7_5(u, start_state)

def __encode_conv_7_5(u, start_state):
    state = start_state.copy()
    p = zeros(len(u))
    for i in range(len(u)):
        new_state = (u[i] + state[0] + state[1]) % 2
        p[i] = (state[1] + new_state) % 2
        state[1] = state[0]
        state[0] = new_state
        #print(state)
    return p, state

def flatten(u):
    return np.array(u).flatten(order='F')

def unflatten(u, l):
    return np.reshape(u, (l, -1), order='F')

def pccc_encode_7_5(u, interleaver):
    enc1, _state1 = encode_conv_7_5(u)
    inp2 = interleave(u, interleaver)
    enc2, _state2 = encode_conv_7_5(inp2)
    #print(u)
    #print(enc1)
    #print(enc2)
    out = flatten([u, enc1, enc2])
    return out

def test_conv_encoder():
    u = zero_pad([1, 0, 0, 0, 1], 15)
    print(u, encode_conv_7_5(u))

def test_interleaver():
    print(np.arange(32))
    interleaver = an_interleaver(32)
    print(interleaver)
    print(deinterleaver(interleaver))
    print(interleave(zero_pad([1,2,3,4], 32), interleaver))
    #print(an_interleaver(128))

def test_pccc_encoder():
    interleaver = an_interleaver(32)
    word1 = zero_pad([1,0,0,1], 32)
    word2 = delay(word1, 1)
    print("WORD1")
    c1 = pccc_encode_7_5(word1, interleaver)
    print(unflatten(c1, 3))
    print(c1)
    print("WORD2")
    c2 = pccc_encode_7_5(word2, interleaver)
    print(unflatten(c2, 3))
    print(c2)

def words_with_weight(weight, k):
    one_locs = combinations(np.arange(k), weight)
    for one_loc in one_locs:
        word = zeros(k)
        word[list(one_loc)] = 1
        yield word

def weight_spectrum_7_5(interleaver, weight):
    k = len(interleaver)
    dmin = np.inf
    nmin = 0
    for word in words_with_weight(weight, k):
        enc = pccc_encode_7_5(word, interleaver)
        wout = sum(enc)
        if wout == dmin:
            nmin += 1
        elif wout < dmin:
            dmin = wout
            nmin = 1
    return dmin, nmin

def test_search_pccc_weightspectrum():
    interleaver = an_interleaver(128)
    #interleaver = rand_interleaver(128)
    #interleaver = s_interleaver(128, 10)
    d2min, n2 = weight_spectrum_7_5(interleaver, 2)
    print("A2", d2min, n2)
    d3min, n3 = weight_spectrum_7_5(interleaver, 3)
    print("A3", d3min, n3)

def is_odd(i): return i % 2
def is_even(i): return not is_odd(i)

# puncture odd: 0 1 2 3 4 -> 0 2 4
# lambda - do we puncture i ?
def odd_puncturer():
    return lambda i: is_odd(i)

# puncture even: 0 1 2 3 4 -> 1 3
def even_puncturer():
    return lambda i: is_even(i)

# do not skip anything
def no_puncturer():
    return lambda _i: False

# fill r with zeros at punctured places
def depuncture(r, k, punc1, punc2):
    out = np.zeros((3, k))
    j = 0
    for i in range(k):
        # information bits are not punctured
        out[0][i] = r[j]
        j += 1
        if not punc1(i):
            out[1][i] = r[j]
            j += 1
        if not punc2(i):
            out[2][i] = r[j]
            j += 1
    #print(out)
    return flatten(out)

def decode_pccc_7_9(r, k, sigma, max_iter, trellis, punc1, punc2):
    sigmasq = sigma ** 2
    perm = rand_interleaver(k)
    inv_perm = deinterleaver(perm)
    r_dep = depuncture(r, k, punc1, punc2)
    [sys, par1, par2] = unflatten(r_dep, 3)

    llrs_21 = np.zeros(k)
    sys_llr = llr_awgn(sys, sigmasq)
    sys_llr_perm = interleave(sys_llr, perm)

    prev_signs = zeros(k)
    llr_threshold = 10
    for i in range(max_iter):
        r1 = flatten(np.array([sys, par1]))
        llrs_1 = bcjr.decode_siso(r1, trellis, sigma, llrs_21)
        llrs_12 = llrs_1 - llrs_21 - sys_llr

        r2 = flatten(np.array([interleave(sys, perm), par2]))
        llrs_12_p = interleave(llrs_12, perm)
        llrs_2 = bcjr.decode_siso(r2, trellis, sigma, llrs_12_p)
        llrs_21_permuted = llrs_2 - llrs_12_p - sys_llr_perm
        llrs_21 = interleave(llrs_21_permuted, inv_perm)
        total_llrs = interleave(llrs_2, inv_perm)
        signs = np.sign(total_llrs)
        if np.array_equal(signs, prev_signs) and np.all(np.abs(total_llrs) > llr_threshold):
            # assume that if signs of llrs are not changing and amplitude is big enough
            # then the decision will not change
            break
        prev_signs = signs
        #print(np.array([llrs_1, total_llrs]))
        #print(np.sign(total_llrs))

    return total_llrs

def simulate_pccc_7_5():
    # BCRJ trellises for component codes
    trellis = build_5_7_trellis()
    punc1 = odd_puncturer()
    punc2 = even_puncturer()

    max_runs = 1e5
    max_iter = 8
    k = 1000
    rate = 1/2
    n = int(k / rate)

    snrDbs = np.arange(1, 4)
    snrDbs = [1, 1.5, 2, 2.5, 3]
    bers = np.zeros(len(snrDbs))
    fers = np.zeros(len(snrDbs))
    i = 0
    for snrDb in snrDbs:
        snr = db2pow(snrDb)
        sigma = bawgn.noise_sigma(snr, rate)

        decode = lambda r : decide_from_llrs(decode_pccc_7_9(r, k, sigma, max_iter, trellis, punc1, punc2))
        ber, fer = simulate_transmission(n, k, max_runs, bawgn.modulate, bawgn.transmit(sigma), decode)

        bers[i] = ber
        fers[i] = fer
        i += 1
        print(snrDb, ber, fer)

    # save to mat file for plotting purposes
    scipy.io.savemat('pccc_7_5_punc_ref', {
        'pccc_7_5_snrs_punc_ref' : snrDbs,
        'pccc_7_5_bers_punc_ref' : bers,
        'pccc_7_5_fers_punc_ref' : fers
        })

def build_5_7_trellis():
    nxt = {}
    numbits = 2
    states = [[0,0],[0,1],[1,0],[1,1]]
    inputs = [0,1]
    for f in states:
        fbit = bcjr.bi2de(f, numbits)
        if fbit not in nxt: nxt[fbit] = {}
        for b in inputs:
            [out], t = __encode_conv_7_5([b], f)
            tbit = bcjr.bi2de(t, numbits)
            nxt[fbit][tbit] = { 'in' : list(bawgn.modulate([b])), 'out' : list(bawgn.modulate([b, out])) }
            #print("FROM:", f, "IN:", b, "OUT:", out, "TO:", t)
    #print(nxt)
    nexts, prevs = bcjr.regular_trellis(nxt)
    return prevs, nexts

def test_build_5_7_trellis():
    prevs, nexts = build_5_7_trellis()
    print("NEXTS")
    [ print("FROM:", f, "IN:", spec['in'], "OUT:", spec['out'], "TO:", t) for f, ts in nexts[0].items() for t, spec in ts.items() ]
    print("PREVS")
    [ print("TO:", f, "IN:", spec['in'], "OUT:", spec['out'], "FROM:", t) for f, ts in prevs[0].items() for t, spec in ts.items() ]

def test_puncturers():
    r = np.array([1,2,3,4,5,6,7,8,9])
    k = 3
    punc1 = no_puncturer()
    punc2 = no_puncturer()
    r_dep = depuncture(r, k, punc1, punc2)
    print(r_dep)

    r = np.array([1,2,3,4,5,6,7,8])
    k = 4
    punc1 = odd_puncturer()
    punc2 = even_puncturer()
    r_dep = depuncture(r, k, punc1, punc2)
    print(r_dep)

def main():
    #test_conv_encoder() # R7.1
    #test_build_5_7_trellis()
    #test_interleaver() # R7.3 interleaver
    #test_pccc_encoder() # R7.3 PCCC encoding
    #test_search_pccc_weightspectrum() # R7.4
    #test_puncturers()
    simulate_pccc_7_5() # P14

if __name__ == '__main__':
    main()
