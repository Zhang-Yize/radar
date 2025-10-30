"""Tests for data structures module"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radar_perception.data_structures import RadarPoint, RadarDetection, TrackedObject


def test_radar_point_creation():
    """Test RadarPoint creation and initialization"""
    point = RadarPoint(
        range=10.0,
        azimuth=0.5,
        elevation=0.1,
        velocity=5.0,
        rcs=10.0,
        snr=15.0
    )
    assert point.range == 10.0
    assert point.azimuth == 0.5
    assert point.velocity == 5.0


def test_radar_point_to_cartesian():
    """Test conversion from spherical to Cartesian coordinates"""
    point = RadarPoint(
        range=10.0,
        azimuth=0.0,
        elevation=0.0,
        velocity=5.0,
        rcs=10.0,
        snr=15.0
    )
    cartesian = point.to_cartesian()
    assert len(cartesian) == 3
    assert np.isclose(cartesian[0], 10.0, rtol=1e-5)
    assert np.isclose(cartesian[1], 0.0, atol=1e-10)
    assert np.isclose(cartesian[2], 0.0, atol=1e-10)


def test_radar_detection_creation():
    """Test RadarDetection creation"""
    points = [
        RadarPoint(10.0, 0.0, 0.0, 5.0, 10.0, 15.0),
        RadarPoint(11.0, 0.1, 0.0, 5.5, 9.0, 14.0)
    ]
    detection = RadarDetection(
        points=points,
        centroid=np.array([10.0, 0.0, 0.0]),
        velocity=np.array([5.0, 0.0, 0.0]),
        rcs=9.5,
        confidence=0.8,
        timestamp=1.0
    )
    assert len(detection.points) == 2
    assert detection.confidence == 0.8
    assert detection.timestamp == 1.0


def test_tracked_object_methods():
    """Test TrackedObject methods"""
    track = TrackedObject(
        track_id=1,
        state=np.array([10.0, 5.0, 0.0, 2.0, 1.0, 0.0]),
        covariance=np.eye(6),
        detections=[],
        age=5,
        hits=10,
        misses=0,
        confidence=0.95
    )
    
    position = track.get_position()
    assert len(position) == 3
    assert position[0] == 10.0
    assert position[1] == 5.0
    
    velocity = track.get_velocity()
    assert len(velocity) == 3
    assert velocity[0] == 2.0
    assert velocity[1] == 1.0


if __name__ == "__main__":
    test_radar_point_creation()
    test_radar_point_to_cartesian()
    test_radar_detection_creation()
    test_tracked_object_methods()
    print("All data structure tests passed!")
