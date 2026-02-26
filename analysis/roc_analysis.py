import numpy as np
from detectors.energy_detector import energy_statistic

def monte_carlo_pd(N, sigma, signal_func, noise_func, gamma, trials=10000):
    detections = 0

    for _ in range(trials):
        s = signal_func(N)
        w = noise_func(N, sigma)
        x = s + w

        T = energy_statistic(x)
        if T > gamma:
            detections += 1

    return detections / trials
