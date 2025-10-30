"""
Radar Perception Algorithm Package

This package provides a comprehensive radar perception system including:
- Signal processing (Range-Doppler processing)
- Object detection (CFAR algorithm)
- Object tracking (Kalman filter)
- Data structures for radar point clouds
"""

from .signal_processing import RangeProcessor, DopplerProcessor
from .detection import CFARDetector
from .tracking import KalmanTracker
from .data_structures import RadarPoint, RadarDetection, TrackedObject

__version__ = "1.0.0"
__all__ = [
    "RangeProcessor",
    "DopplerProcessor",
    "CFARDetector",
    "KalmanTracker",
    "RadarPoint",
    "RadarDetection",
    "TrackedObject",
]
