import numpy as np
from scipy.stats import norm

def theoretical_pd(N, sigma, snr_linear, Pfa):
    """
    Gaussian approximation-based theoretical Pd
    """
    signal_power = snr_linear * sigma**2
    mean1 = N * (sigma**2 + signal_power)
    var1 = 2 * N * (sigma**2 + signal_power)**2
    std1 = np.sqrt(var1)

    mean0 = N * sigma**2
    var0 = 2 * N * sigma**4
    std0 = np.sqrt(var0)

    gamma = mean0 + std0 * norm.ppf(1 - Pfa)

    Pd = 1 - norm.cdf((gamma - mean1) / std1)
    return Pd
