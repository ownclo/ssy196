# Exhaustive search maximum likelihood decoder
# i.e. minimum Euclidean-distance decoder
import numpy as np
import bawgn
import misc

def all_codewords():
    return np.array(
           [[0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1]])

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def min_by(pred, arr):
    return arr[np.argmin(np.apply_along_axis(pred, 1, arr))]

def decode_ml(codewords, received):
    return min_by(lambda cw : euclidean_distance(bawgn.modulate(cw), received), codewords)

def main():
    H = [[1, 0, 0, 1, 1, 0, 1],
         [0, 1, 0, 1, 0, 1, 1],
         [0, 0, 1, 0, 1, 1, 1]]

    Ha = np.array(H)

    G = [[1, 1, 0, 1, 0, 0, 0],
         [1, 0, 1, 0, 1, 0, 0],
         [0, 1, 1, 0, 0, 1, 0],
         [1, 1, 1, 0, 0, 0, 1]]

    cws = all_codewords()

    for cw in cws:
        print(misc.is_codeword(cw, Ha))
        #print(cw, decode_ml(cws, bawgn.modulate(cw)))


if __name__ == '__main__':
    main()
