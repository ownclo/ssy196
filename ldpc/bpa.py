"""
Beleif propagation on a bipartite graph.
Primarily for LDPC codes, hence the notation.
VN stands for 'Variable Node', CN - for 'Check Node'
"""
import numpy as np
from copy import deepcopy

# extrinsic information principle. For each i, include the incoming
# message from that i.
def slice_dict(adict):
    return { i : [value for key, value in adict.items() if key != i] for i in adict.keys() }

# no optimization of pfunc - copy each time
# nd * (nd - 1) operations.
def bpa_process(inbox, pfunc):
    return { i : pfunc(inbox_slice) for i, inbox_slice in slice_dict(inbox).items() }

def bpa_stage(inboxes, outboxes, pfunc):
    for i, inbox in inboxes.items():
        outbox = bpa_process(inbox, pfunc)
        for dst, msg in outbox.items():
            outboxes[i][dst] = msg

def flipcopy(d1, d2):
    for k1, v1 in d1.items():
        for k2, v2 in v1.items():
            d2[k2][k1] = v2

def flipinit(d1):
    d2 = {}
    for k1, v1 in d1.items():
        for k2, v2 in v1.items():
            if (k2 not in d2):
                d2[k2] = {}
            d2[k2][k1] = v2
    return d2

# one round of information exchange in a bipartite graph
def bpa_round(in1, out1, in2, out2, pfunc1, pfunc2):
    bpa_stage(in1, out1, pfunc1)
    flipcopy(out1, in2)
    bpa_stage(in2, out2, pfunc2)
    flipcopy(out2, in1)

def bpa_loop(in1, out1, in2, out2, pfunc1, pfunc2, max_iter, need_stop):
    for i in range(max_iter):
        if (need_stop(in1)):
            break
        bpa_round(in1, out1, in2, out2, pfunc1, pfunc2)

def bpa_init(H):
    incn = bpa_init_check(H)
    outcn = deepcopy(incn)
    invn = flipinit(outcn)
    outvn = deepcopy(invn)
    return incn, outcn, invn, outvn

# returns input dict for CNs, initialized by 0
def bpa_init_check(H):
    incheck = {}
    for i in range(len(H)):
        if (i not in incheck):
            incheck[i] = {}
        for j in range(len(H[i])):
            if (H[i][j] != 0):
                incheck[i][j] = 0
    return incheck



def main():
    print("Hello, World!")

if __name__ == '__main__':
    main()
