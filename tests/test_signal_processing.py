"""Tests for signal processing module"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radar_perception.signal_processing import RangeProcessor, DopplerProcessor, apply_cfar_2d


def test_range_processor_initialization():
    """Test RangeProcessor initialization"""
    processor = RangeProcessor(
        num_samples=256,
        sample_rate=5e6,
        chirp_bandwidth=100e6
    )
    assert processor.num_samples == 256
    assert processor.sample_rate == 5e6
    assert processor.range_resolution > 0


def test_range_processor_process():
    """Test range processing"""
    processor = RangeProcessor(
        num_samples=256,
        sample_rate=5e6,
        chirp_bandwidth=100e6
    )
    
    # Create synthetic ADC data
    num_chirps = 128
    adc_data = np.random.randn(num_chirps, 256) + 1j * np.random.randn(num_chirps, 256)
    
    # Process
    range_profile = processor.process(adc_data)
    
    assert range_profile.shape == (num_chirps, 256)
    assert np.all(range_profile >= 0)  # Magnitude is non-negative


def test_range_processor_get_range_bins():
    """Test range bin calculation"""
    processor = RangeProcessor(
        num_samples=256,
        sample_rate=5e6,
        chirp_bandwidth=100e6
    )
    
    range_bins = processor.get_range_bins()
    assert len(range_bins) == 256
    assert range_bins[0] == 0
    assert range_bins[-1] > range_bins[0]


def test_doppler_processor_initialization():
    """Test DopplerProcessor initialization"""
    processor = DopplerProcessor(
        num_chirps=128,
        chirp_period=50e-6,
        wavelength=0.004
    )
    assert processor.num_chirps == 128
    assert processor.chirp_period == 50e-6
    assert processor.velocity_resolution > 0


def test_doppler_processor_process():
    """Test Doppler processing"""
    processor = DopplerProcessor(
        num_chirps=128,
        chirp_period=50e-6,
        wavelength=0.004
    )
    
    # Create synthetic range profile
    range_profile = np.random.randn(128, 256)
    
    # Process
    range_doppler_map = processor.process(range_profile)
    
    assert range_doppler_map.shape == (128, 256)
    assert np.all(range_doppler_map >= 0)  # Power is non-negative


def test_doppler_processor_get_velocity_bins():
    """Test velocity bin calculation"""
    processor = DopplerProcessor(
        num_chirps=128,
        chirp_period=50e-6,
        wavelength=0.004
    )
    
    velocity_bins = processor.get_velocity_bins()
    assert len(velocity_bins) == 128
    assert velocity_bins[0] < 0  # Negative velocities
    assert velocity_bins[-1] > 0  # Positive velocities


def test_cfar_2d_basic():
    """Test 2D CFAR detection"""
    # Create synthetic data with target
    data = np.random.randn(100, 100) * 0.5 + 1.0
    
    # Add strong target
    data[50, 50] = 100.0
    
    # Apply CFAR
    detections = apply_cfar_2d(
        data,
        guard_cells=(2, 2),
        training_cells=(4, 4),
        pfa=1e-6
    )
    
    assert detections.shape == data.shape
    assert detections.dtype == bool
    assert detections[50, 50] == True  # Target should be detected


def test_cfar_2d_no_detection():
    """Test CFAR with no targets"""
    # Create noise-only data
    data = np.random.randn(100, 100) * 0.5 + 1.0
    
    # Apply CFAR
    detections = apply_cfar_2d(
        data,
        guard_cells=(2, 2),
        training_cells=(4, 4),
        pfa=1e-6
    )
    
    # Should have very few detections in noise
    assert np.sum(detections) < 10


if __name__ == "__main__":
    test_range_processor_initialization()
    test_range_processor_process()
    test_range_processor_get_range_bins()
    test_doppler_processor_initialization()
    test_doppler_processor_process()
    test_doppler_processor_get_velocity_bins()
    test_cfar_2d_basic()
    test_cfar_2d_no_detection()
    print("All signal processing tests passed!")
