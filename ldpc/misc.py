import numpy as np

def syndrome(r, H):
    return np.mod(r.dot(H.T), 2)

def is_codeword(r, H):
    return not np.any(syndrome(r, H))
