import numpy as np
from scipy.stats import norm

def energy_statistic(x):
    return np.sum(np.abs(x)**2)

def threshold_from_pfa(N, sigma, Pfa):
    """
    Gaussian approximation of chi-square statistic
    """
    mean = N * sigma**2
    variance = 2 * N * sigma**4
    std = np.sqrt(variance)

    gamma = mean + std * norm.ppf(1 - Pfa)
    return gamma
