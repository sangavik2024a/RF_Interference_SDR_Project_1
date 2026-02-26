import numpy as np

def generate_complex_awgn(N, sigma):
    """
    Generate complex AWGN samples:
    w[n] ~ CN(0, sigma^2)
    """
    noise_real = np.random.normal(0, sigma/np.sqrt(2), N)
    noise_imag = np.random.normal(0, sigma/np.sqrt(2), N)
    return noise_real + 1j * noise_imag
