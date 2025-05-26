
import numpy as np

def add_awgn_noise(signal, snr_db):
    """Add AWGN noise to a signal."""
    snr_linear = 10**(snr_db / 10)
    power_signal = np.mean(np.abs(signal)**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * (np.random.normal(0, 1, len(signal)) + 1j*np.random.normal(0, 1, len(signal))) / np.sqrt(2)
    return signal + noise
