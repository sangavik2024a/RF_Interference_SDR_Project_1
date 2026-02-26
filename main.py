import numpy as np
import matplotlib.pyplot as plt

from core.signal_models import generate_qpsk_signal
from core.noise_models import generate_complex_awgn
from detectors.energy_detector import threshold_from_pfa
from analysis.roc_analysis import monte_carlo_pd
from analysis.theoretical_validation import theoretical_pd

# Parameters
N = 128
sigma = 1
Pfa = 0.05
snr_db_range = np.arange(-10, 11, 2)
trials = 10000

gamma = threshold_from_pfa(N, sigma, Pfa)

pd_sim = []
pd_theory = []

for snr_db in snr_db_range:
    snr_linear = 10**(snr_db/10)

    def signal_scaled(N):
        amplitude = np.sqrt(snr_linear * sigma**2)
        return generate_qpsk_signal(N, amplitude)

    pd = monte_carlo_pd(
        N,
        sigma,
        signal_scaled,
        generate_complex_awgn,
        gamma,
        trials
    )

    pd_sim.append(pd)
    pd_theory.append(
        theoretical_pd(N, sigma, snr_linear, Pfa)
    )

plt.plot(snr_db_range, pd_sim, 'o-', label="Simulated Pd")
plt.plot(snr_db_range, pd_theory, '--', label="Theoretical Pd")
plt.xlabel("SNR (dB)")
plt.ylabel("Probability of Detection (Pd)")
plt.title("Theoretical vs Simulated Detection Performance")
plt.legend()
plt.grid()
plt.show()
