import numpy as np

def modulate(word):
    return np.array([1 if b == 0 else -1 for b in word])

def demodulate_hard(word):
    return np.array([0 if s >= 0 else 1 for s in word])

def transmit(sigma):
    return lambda word : _transmit(word, sigma)

def _transmit(word, sigma):
    return np.add(word, np.random.normal(0.0, sigma, len(word)))

def noise_sigma(eb_n0, code_rate):
    return 1.0 / np.sqrt(2 * eb_n0 * code_rate)

def main():
    sigma = 0.01
    print(modulate([0, 1, 0, 1, 0]))
    print(transmit(modulate([0, 1, 0, 1, 0]), sigma))

if __name__ == '__main__':
    main()
