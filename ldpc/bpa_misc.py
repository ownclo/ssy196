def init_cn_inbox(r, incn, rfun):
    for i in range(len(r)):
        for cn_idx, cn_inbox_dict in incn.items():
            if i in cn_inbox_dict:
                cn_inbox_dict[i] = rfun(r[i])
