import numpy as np

def sliding_window_energy(x, window_size):
    energies = []
    for i in range(len(x) - window_size + 1):
        window = x[i:i+window_size]
        energies.append(np.sum(np.abs(window)**2))
    return np.array(energies)
