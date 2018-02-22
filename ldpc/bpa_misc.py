import numpy as np

def init_cn_inbox(r, incn, rfun):
    for i in range(len(r)):
        for cn_idx, cn_inbox_dict in incn.items():
            if i in cn_inbox_dict:
                cn_inbox_dict[i] = rfun(r[i])

def decide_from_llrs(llrs):
    return np.array([decide_from_llr(llr) for llr in llrs])

def decide_from_llr(llr):
    if llr == 0: return 0 if random() > 0 else 1
    else: return 1 if llr < 0 else 0
