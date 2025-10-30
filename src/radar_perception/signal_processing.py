"""
Radar signal processing module

Implements range-Doppler processing for radar signals
"""
import numpy as np
from typing import Tuple


class RangeProcessor:
    """Processes radar signals to extract range information"""
    
    def __init__(self, num_samples: int, sample_rate: float, chirp_bandwidth: float):
        """
        Initialize range processor
        
        Args:
            num_samples: Number of ADC samples per chirp
            sample_rate: ADC sampling rate in Hz
            chirp_bandwidth: Chirp bandwidth in Hz
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.chirp_bandwidth = chirp_bandwidth
        self.range_resolution = 3e8 / (2 * chirp_bandwidth)  # c / (2 * B)
        
    def process(self, adc_data: np.ndarray) -> np.ndarray:
        """
        Process ADC data to generate range profile
        
        Args:
            adc_data: Complex ADC samples, shape (num_chirps, num_samples)
            
        Returns:
            Range profile, shape (num_chirps, num_range_bins)
        """
        # Apply window to reduce sidelobes
        window = np.hanning(self.num_samples)
        windowed_data = adc_data * window[np.newaxis, :]
        
        # Perform FFT along range dimension
        range_fft = np.fft.fft(windowed_data, axis=1)
        
        # Take magnitude
        range_profile = np.abs(range_fft)
        
        return range_profile
    
    def get_range_bins(self) -> np.ndarray:
        """Get range bin values in meters"""
        return np.arange(self.num_samples) * self.range_resolution


class DopplerProcessor:
    """Processes radar signals to extract velocity information"""
    
    def __init__(self, num_chirps: int, chirp_period: float, wavelength: float):
        """
        Initialize Doppler processor
        
        Args:
            num_chirps: Number of chirps in a frame
            chirp_period: Time between chirps in seconds
            wavelength: Radar wavelength in meters
        """
        self.num_chirps = num_chirps
        self.chirp_period = chirp_period
        self.wavelength = wavelength
        self.velocity_resolution = wavelength / (2 * num_chirps * chirp_period)
        
    def process(self, range_profile: np.ndarray) -> np.ndarray:
        """
        Process range profile to generate range-Doppler map
        
        Args:
            range_profile: Range profile data, shape (num_chirps, num_range_bins)
            
        Returns:
            Range-Doppler map, shape (num_doppler_bins, num_range_bins)
        """
        # Apply window to reduce sidelobes
        window = np.hanning(self.num_chirps)
        windowed_profile = range_profile * window[:, np.newaxis]
        
        # Perform FFT along Doppler dimension
        doppler_fft = np.fft.fftshift(np.fft.fft(windowed_profile, axis=0), axes=0)
        
        # Take magnitude squared (power)
        range_doppler_map = np.abs(doppler_fft) ** 2
        
        return range_doppler_map
    
    def get_velocity_bins(self) -> np.ndarray:
        """Get velocity bin values in m/s"""
        max_velocity = self.wavelength / (4 * self.chirp_period)
        return np.linspace(-max_velocity, max_velocity, self.num_chirps)


def apply_cfar_2d(data: np.ndarray, guard_cells: Tuple[int, int], 
                  training_cells: Tuple[int, int], pfa: float = 1e-6) -> np.ndarray:
    """
    Apply 2D CFAR (Constant False Alarm Rate) detection
    
    Args:
        data: 2D array of signal power
        guard_cells: (range_guard, doppler_guard) guard cell sizes
        training_cells: (range_training, doppler_training) training cell sizes
        pfa: Probability of false alarm
        
    Returns:
        Binary detection mask
    """
    guard_r, guard_d = guard_cells
    train_r, train_d = training_cells
    
    # Calculate threshold multiplier based on desired PFA
    num_training = (2 * (train_r + guard_r) + 1) * (2 * (train_d + guard_d) + 1) - \
                   (2 * guard_r + 1) * (2 * guard_d + 1)
    alpha = num_training * (pfa ** (-1.0 / num_training) - 1)
    
    detections = np.zeros_like(data, dtype=bool)
    
    # Iterate through each cell
    for i in range(train_r + guard_r, data.shape[0] - train_r - guard_r):
        for j in range(train_d + guard_d, data.shape[1] - train_d - guard_d):
            # Extract training region
            r_start = i - train_r - guard_r
            r_end = i + train_r + guard_r + 1
            d_start = j - train_d - guard_d
            d_end = j + train_d + guard_d + 1
            
            training_region = data[r_start:r_end, d_start:d_end].copy()
            
            # Zero out guard cells and CUT (Cell Under Test)
            g_r_start = train_r
            g_r_end = train_r + 2 * guard_r + 1
            g_d_start = train_d
            g_d_end = train_d + 2 * guard_d + 1
            training_region[g_r_start:g_r_end, g_d_start:g_d_end] = 0
            
            # Calculate noise level
            noise_level = np.sum(training_region) / num_training
            
            # Calculate threshold
            threshold = alpha * noise_level
            
            # Compare CUT with threshold
            if data[i, j] > threshold:
                detections[i, j] = True
    
    return detections
