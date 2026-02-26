import numpy as np

def covariance_statistic(x):
    """
    Use first-lag correlation magnitude.
    """
    N = len(x)
    R1 = np.sum(x[1:] * np.conj(x[:-1])) / (N - 1)
    return np.abs(R1)
