from bpa import *

def main():
    H = [[1, 1, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [0, 1, 1, 0, 1]]
    r = [1, 1, 1, 1, 1]

    incn, outcn, invn, outvn = bpa_init(H)

    pfuncn = lambda xs : sum(xs) % 2
    pfunvn = lambda xs : 
    print(incn)

if __name__ == '__main__':
    main()
