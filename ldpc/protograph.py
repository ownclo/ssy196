import numpy as np

np.set_printoptions(linewidth=150)

# replace all ones in H_proto with randomly choosen elements
# of perm_matrices
def gen_from_proto(H_proto, perm_matrices):
    p_h, p_w = perm_shape = perm_matrices[0].shape
    H_res = np.zeros(np.multiply(H_proto.shape, perm_shape))

    for i in range(H_proto.shape[0]):
        for j in range(H_proto.shape[1]):
            if H_proto[i][j] != 0:
                H_res[i * p_h : (i+1) * p_h , j * p_w : (j + 1) * p_w] = perm_matrices[np.random.randint(len(perm_matrices))]

    return H_res

def num_4_cycles(H):
    n = 0
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):  # for positions of top-left 1s
            if H[i][j] != 0:
                for i1 in np.arange(i + 1, H.shape[0]):
                    if H[i1][j] != 0:  # corresponding bottom-left is non-zero
                        for j1 in np.arange(j + 1, H.shape[1]):
                            if H[i1][j1] != 0 and H[i][j1] != 0:
                                n += 1
    return n

def main():
    H = [[1, 0, 1, 0, 1, 0, 1],
         [0, 1, 1, 0, 0, 1, 1],
         [0, 0, 0, 1, 1, 1, 1]]
    H_proto = np.array(H)

    p1 = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]
    p1 = np.array(p1)

    p2 = [[0, 1, 0],
          [0, 0, 1],
          [1, 0, 0]]
    p2 = np.array(p2)

    p3 = [[0, 0, 1],
          [1, 0, 0],
          [0, 1, 0]]
    p3 = np.array(p3)

    H_exp = gen_from_proto(H_proto, [p1, p2, p3])
    n1 = num_4_cycles(H_proto)
    n2 = num_4_cycles(H_exp)
    print(H_exp)
    print(n1)
    print(n2)


if __name__ == '__main__':
    main()
