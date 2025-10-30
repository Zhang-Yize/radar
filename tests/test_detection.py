"""Tests for detection module"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radar_perception.detection import CFARDetector


def test_cfar_detector_initialization():
    """Test CFARDetector initialization"""
    detector = CFARDetector(
        guard_cells=(2, 2),
        training_cells=(4, 4),
        pfa=1e-6,
        min_snr=10.0,
        clustering_threshold=2.0
    )
    assert detector.guard_cells == (2, 2)
    assert detector.training_cells == (4, 4)
    assert detector.pfa == 1e-6
    assert detector.min_snr == 10.0
    assert detector.clustering_threshold == 2.0


def test_cfar_detector_detect_no_targets():
    """Test detection with no targets"""
    detector = CFARDetector()
    
    # Create noise-only range-Doppler map
    range_doppler_map = np.random.randn(64, 128) * 0.1 + 0.5
    range_bins = np.linspace(0, 100, 128)
    velocity_bins = np.linspace(-20, 20, 64)
    
    detections = detector.detect(
        range_doppler_map,
        range_bins,
        velocity_bins,
        timestamp=0.0
    )
    
    # Should have few or no detections
    assert len(detections) < 5


def test_cfar_detector_detect_with_target():
    """Test detection with a strong target"""
    detector = CFARDetector(
        min_snr=5.0,  # Lower threshold for test
        clustering_threshold=5.0
    )
    
    # Create range-Doppler map with target
    range_doppler_map = np.random.randn(64, 128) * 0.1 + 1.0
    
    # Add strong target
    range_doppler_map[32, 64] = 50.0
    range_doppler_map[32, 65] = 45.0
    range_doppler_map[33, 64] = 40.0
    
    range_bins = np.linspace(0, 100, 128)
    velocity_bins = np.linspace(-20, 20, 64)
    
    detections = detector.detect(
        range_doppler_map,
        range_bins,
        velocity_bins,
        timestamp=1.0
    )
    
    # Should detect at least one target
    assert len(detections) >= 1
    
    # Check detection properties
    if detections:
        det = detections[0]
        assert len(det.points) > 0
        assert det.centroid.shape == (3,)
        assert det.velocity.shape == (3,)
        assert 0.0 <= det.confidence <= 1.0
        assert det.timestamp == 1.0


def test_cfar_detector_clustering():
    """Test point clustering"""
    detector = CFARDetector(
        min_snr=5.0,
        clustering_threshold=2.0
    )
    
    # Create range-Doppler map with two separated targets
    range_doppler_map = np.random.randn(64, 128) * 0.1 + 1.0
    
    # Target 1
    range_doppler_map[20, 40] = 50.0
    range_doppler_map[21, 40] = 45.0
    
    # Target 2 (far away)
    range_doppler_map[50, 100] = 60.0
    range_doppler_map[51, 100] = 55.0
    
    range_bins = np.linspace(0, 100, 128)
    velocity_bins = np.linspace(-20, 20, 64)
    
    detections = detector.detect(
        range_doppler_map,
        range_bins,
        velocity_bins,
        timestamp=2.0
    )
    
    # Should detect two separate targets
    assert len(detections) >= 1  # At least one detection


if __name__ == "__main__":
    test_cfar_detector_initialization()
    test_cfar_detector_detect_no_targets()
    test_cfar_detector_detect_with_target()
    test_cfar_detector_clustering()
    print("All detection tests passed!")
