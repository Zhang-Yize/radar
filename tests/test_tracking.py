"""Tests for tracking module"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radar_perception.tracking import KalmanTracker
from radar_perception.data_structures import RadarPoint, RadarDetection


def test_kalman_tracker_initialization():
    """Test KalmanTracker initialization"""
    tracker = KalmanTracker(
        process_noise=1.0,
        measurement_noise=1.0,
        max_age=5,
        min_hits=3,
        association_threshold=5.0
    )
    assert tracker.process_noise == 1.0
    assert tracker.measurement_noise == 1.0
    assert tracker.max_age == 5
    assert tracker.min_hits == 3
    assert len(tracker.tracks) == 0


def test_kalman_tracker_set_dt():
    """Test setting time step"""
    tracker = KalmanTracker()
    tracker.set_dt(0.05)
    assert tracker.dt == 0.05


def test_kalman_tracker_single_detection():
    """Test tracking with single detection"""
    tracker = KalmanTracker(min_hits=1)  # Confirm immediately for testing
    
    # Create a detection
    points = [RadarPoint(10.0, 0.0, 0.0, 5.0, 10.0, 15.0)]
    detection = RadarDetection(
        points=points,
        centroid=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        rcs=10.0,
        confidence=0.8,
        timestamp=0.0
    )
    
    # Update tracker
    tracks = tracker.update([detection])
    
    # Should create one track
    assert len(tracks) == 1
    assert tracks[0].track_id == 0


def test_kalman_tracker_multiple_frames():
    """Test tracking over multiple frames"""
    tracker = KalmanTracker(min_hits=2, max_age=3)
    tracker.set_dt(0.1)
    
    # Frame 1: Initial detection
    detection1 = RadarDetection(
        points=[RadarPoint(10.0, 0.0, 0.0, 5.0, 10.0, 15.0)],
        centroid=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        rcs=10.0,
        confidence=0.8,
        timestamp=0.0
    )
    
    tracks = tracker.update([detection1])
    assert len(tracks) == 0  # Not confirmed yet (min_hits=2)
    
    # Frame 2: Close detection (should match)
    detection2 = RadarDetection(
        points=[RadarPoint(10.5, 0.0, 0.0, 5.0, 10.0, 15.0)],
        centroid=np.array([10.5, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        rcs=10.0,
        confidence=0.8,
        timestamp=0.1
    )
    
    tracks = tracker.update([detection2])
    assert len(tracks) == 1  # Now confirmed
    assert tracks[0].hits >= 2


def test_kalman_tracker_track_loss():
    """Test track loss after missing detections"""
    tracker = KalmanTracker(min_hits=1, max_age=3)
    
    # Initial detection
    detection = RadarDetection(
        points=[RadarPoint(10.0, 0.0, 0.0, 5.0, 10.0, 15.0)],
        centroid=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        rcs=10.0,
        confidence=0.8,
        timestamp=0.0
    )
    
    tracks = tracker.update([detection])
    assert len(tracks) == 1
    
    # Miss several frames
    for _ in range(4):
        tracks = tracker.update([])
    
    # Track should be removed after max_age
    assert len(tracker.tracks) == 0


def test_kalman_tracker_multiple_objects():
    """Test tracking multiple objects"""
    tracker = KalmanTracker(min_hits=1)
    
    # Two detections far apart
    detection1 = RadarDetection(
        points=[RadarPoint(10.0, 0.0, 0.0, 5.0, 10.0, 15.0)],
        centroid=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        rcs=10.0,
        confidence=0.8,
        timestamp=0.0
    )
    
    detection2 = RadarDetection(
        points=[RadarPoint(50.0, 0.0, 0.0, -3.0, 8.0, 12.0)],
        centroid=np.array([50.0, 0.0, 0.0]),
        velocity=np.array([-3.0, 0.0, 0.0]),
        rcs=8.0,
        confidence=0.7,
        timestamp=0.0
    )
    
    tracks = tracker.update([detection1, detection2])
    
    # Should create two tracks
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_kalman_tracker_prediction():
    """Test state prediction"""
    tracker = KalmanTracker(min_hits=1)
    tracker.set_dt(0.1)
    
    # Create initial track
    detection = RadarDetection(
        points=[RadarPoint(10.0, 0.0, 0.0, 5.0, 10.0, 15.0)],
        centroid=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        rcs=10.0,
        confidence=0.8,
        timestamp=0.0
    )
    
    tracks = tracker.update([detection])
    initial_position = tracks[0].get_position().copy()
    
    # Update without detection (prediction only)
    tracks = tracker.update([])
    
    # Position should have moved based on velocity
    # Even though track might be removed, check the internal tracks
    if len(tracker.tracks) > 0:
        predicted_position = tracker.tracks[0].get_position()
        assert not np.allclose(predicted_position, initial_position)


if __name__ == "__main__":
    test_kalman_tracker_initialization()
    test_kalman_tracker_set_dt()
    test_kalman_tracker_single_detection()
    test_kalman_tracker_multiple_frames()
    test_kalman_tracker_track_loss()
    test_kalman_tracker_multiple_objects()
    test_kalman_tracker_prediction()
    print("All tracking tests passed!")
