
import numpy as np

def add_awgn_noise(signal, snr_db):
    """Add AWGN noise to a real-valued signal."""
    snr_linear = 10**(snr_db / 10)
    power_signal = np.mean(signal**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.normal(0, 1, len(signal))
    return signal + noise
