import numpy as np

def generate_qpsk_signal(N, amplitude=1.0):
    """
    Generate random QPSK signal.
    """
    bits = np.random.randint(0, 4, N)
    mapping = {
        0: 1 + 1j,
        1: -1 + 1j,
        2: -1 - 1j,
        3: 1 - 1j
    }
    symbols = np.array([mapping[b] for b in bits])
    return amplitude * symbols / np.sqrt(2)
