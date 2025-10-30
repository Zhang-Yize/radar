"""
Data structures for radar perception system
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class RadarPoint:
    """Represents a single radar measurement point"""
    range: float  # Distance in meters
    azimuth: float  # Angle in radians
    elevation: float  # Elevation angle in radians
    velocity: float  # Radial velocity in m/s
    rcs: float  # Radar cross section in dBsm
    snr: float  # Signal-to-noise ratio in dB
    
    def to_cartesian(self) -> np.ndarray:
        """Convert spherical coordinates to Cartesian"""
        x = self.range * np.cos(self.elevation) * np.cos(self.azimuth)
        y = self.range * np.cos(self.elevation) * np.sin(self.azimuth)
        z = self.range * np.sin(self.elevation)
        return np.array([x, y, z])


@dataclass
class RadarDetection:
    """Represents a detected object from radar data"""
    points: List[RadarPoint]
    centroid: np.ndarray  # 3D position (x, y, z)
    velocity: np.ndarray  # 3D velocity vector
    rcs: float  # Average radar cross section
    confidence: float  # Detection confidence [0, 1]
    timestamp: float  # Time of detection


@dataclass
class TrackedObject:
    """Represents a tracked object over time"""
    track_id: int
    state: np.ndarray  # State vector [x, y, z, vx, vy, vz]
    covariance: np.ndarray  # State covariance matrix
    detections: List[RadarDetection]  # History of detections
    age: int  # Number of frames since first detection
    hits: int  # Number of successful matches
    misses: int  # Number of consecutive missed detections
    confidence: float  # Overall tracking confidence
    
    def get_position(self) -> np.ndarray:
        """Get current 3D position"""
        return self.state[:3]
    
    def get_velocity(self) -> np.ndarray:
        """Get current 3D velocity"""
        return self.state[3:6]
