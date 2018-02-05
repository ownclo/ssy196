import numpy as np

def syndrome(r, H):
    return np.mod(r.dot(H.T), 2)

def nf(r, H):
    """
    the number of failed constraints that a bit is involved in
    return - 1-d array index of bit -> number of failed constraints
    """
    return syndrome(r,H).dot(H)

def flip(r, t, failed):
    """
    Flip bits in r which are involved in at least t failed
    constraints, according to _failed_ vector.
    """
    ret = r.copy()
    ret[failed >= t] += 1
    return np.mod(ret, 2)

def bitflip_round(r, H, t):
    nfail = nf(r, H)
    return flip(r, t, nfail)

def bitflip_decode(r, H, t, maxIter):
    rhat = r
    rprev = rhat
    for i in range(maxIter):
        rhat = bitflip_round(rhat, H, t)
        if (np.array_equal(rhat, rprev)):
            break
        rprev = rhat
    return rhat

def main():
    n = 5
    k = 2
    t = 2  # bitflip threshold (see doc for flip)
    H_arr = [[1, 1, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 1, 1, 0, 1]]
    H = np.array(H_arr)

    r = [1, 1, 1, 1, 1]
    r = np.array(r)

    print(r)
    print(bitflip_decode(r, H, 2, 100))


if __name__ == '__main__':
    main()
