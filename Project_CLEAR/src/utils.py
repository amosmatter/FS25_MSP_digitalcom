
import numpy as np
import matplotlib.pyplot as plt

def add_awgn_noise(signal, snr_db):
    """Additive White Gaussian Noise (AWGN) to a signal."""
    snr_linear = 10**(snr_db / 10)
    power_signal = np.mean(np.abs(signal)**2)
    power_noise = power_signal / snr_linear
    noise = np.sqrt(power_noise) * np.random.normal(0, 1, len(signal))
    return signal + noise

def plot_psd(signal, title='PSD', Fs=1.0):
    """Plot Power Spectral Density (PSD) using FFT."""
    N = len(signal)
    f = np.fft.fftfreq(N, d=1/Fs)
    S = np.abs(np.fft.fft(signal))**2 / N
    f_shift = np.fft.fftshift(f)
    S_shift = np.fft.fftshift(S)
    plt.figure(figsize=(8, 4))
    plt.plot(f_shift, 10*np.log10(S_shift + 1e-12))
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('PSD (dB)')
    plt.grid()
    plt.show()
