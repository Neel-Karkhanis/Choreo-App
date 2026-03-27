import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
from scipy.ndimage import median_filter


def load_audio(filepath):
    sample_rate, data = wavfile.read(filepath)
    if data.ndim == 2:
        data = np.mean(data, 1)
    return sample_rate, data/32767

def stft(audio, window_size=2048, hop_length=512):

    # separating the data into managable chunks
    window = []
    for i in range(0, len(audio), hop_length):
        chunk = audio[i:i+window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))

        # hanning each chunk to smoothen out
        chunk *= np.hanning(window_size)
        # getting the Real Fast Fourier Transform on each chunk
        chunk = np.fft.rfft(chunk)

        window.append(chunk)
    window = np.array(window)
    window = window.T
    return window

def compute_masks(fourier_transform, harm_size=31, perc_size=31):
    mag = np.abs(fourier_transform)
    
    horizontal = median_filter(mag, size=(1, harm_size))
    vertical = median_filter(mag, size=(perc_size, 1))
    
    return horizontal, vertical

def build_masks(horizontal, vertical):
    harmonic_mask = horizontal / (horizontal + vertical + 1e-8)
    percussive_mask = vertical / (vertical + horizontal + 1e-8)
    return harmonic_mask, percussive_mask

def apply_masks(harmonic_mask, percussive_mask, fourier_transform):
    harmonic_spectrogram = harmonic_mask * fourier_transform
    percussive_spectrogram = percussive_mask * fourier_transform
    return harmonic_spectrogram, percussive_spectrogram






        



