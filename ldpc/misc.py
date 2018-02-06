import numpy as np

def syndrome(r, H):
    return np.mod(r.dot(H.T), 2)

def is_codeword(r, H):
    return not np.any(syndrome(r, H))


def pow2db(x):
    return 10 * np.log10(x)


def db2pow(xdb):
    return 10.**(xdb/10.)


def last_k(word, k):
    return word[-k:]
