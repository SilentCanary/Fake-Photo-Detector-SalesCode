import numpy as np
import cv2 


def extract_fft_features(gray:np.ndarray)->dict:
    # FFT-based features for detecting periodic screen/grid artifacts.
    #    Screen recaptures often create artificial high-frequency spikes because display pixels interfere with camera sensor sampling.

    gray=gray.astype(np.float32)/255.0

    window=np.outer(np.hanning(gray.shape[0]),np.hanning(gray.shape[1]))
    gray_windowed=gray*window

    fft=np.fft.fft2(gray_windowed)
    fft_shift=np.fft.fftshift(fft)

    magnitude=np.abs(fft_shift)
    log_magnitude=np.log1p(magnitude)

    h,w=log_magnitude.shape
    cy,cx=h//2,w//2

    yy,xx=np.ogrid[:h,:w]
    radius=np.sqrt((yy-cy)**2 + (xx-cx)**2)

    low_mask=radius<min(h,w)*0.06
    high_mask=radius>min(h,w)*0.20

    valid=~low_mask
    high=high_mask
    valid_values = log_magnitude[valid]
    high_values = log_magnitude[high]

    mean_val = float(np.mean(valid_values))
    std_val = float(np.std(valid_values) + 1e-8)

    # Spike count: unusually strong frequency points
    z = (log_magnitude - mean_val) / std_val
    spike_count = int(np.sum((z > 4.0) & valid))
    spike_ratio = float(spike_count / np.sum(valid))

    # Energy ratio in high-frequency region
    high_energy = float(np.sum(log_magnitude[high]))
    total_energy = float(np.sum(log_magnitude[valid]) + 1e-8)
    high_freq_energy_ratio = high_energy / total_energy

    # Directional frequency strength
    vertical_band = log_magnitude[:, cx - 2:cx + 3]
    horizontal_band = log_magnitude[cy - 2:cy + 3, :]

    vertical_strength = float(np.mean(vertical_band))
    horizontal_strength = float(np.mean(horizontal_band))

    return {
        "fft_mean": mean_val,
        "fft_std": std_val,
        "fft_spike_count": spike_count,
        "fft_spike_ratio": spike_ratio,
        "fft_high_freq_energy_ratio": high_freq_energy_ratio,
        "fft_vertical_strength": vertical_strength,
        "fft_horizontal_strength": horizontal_strength,
    }