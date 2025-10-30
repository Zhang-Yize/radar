"""
Object tracking module using Kalman filter
"""
import numpy as np
from typing import List, Optional
from .data_structures import RadarDetection, TrackedObject


class KalmanTracker:
    """Implements multi-object tracking using Kalman filter"""
    
    def __init__(self,
                 process_noise: float = 1.0,
                 measurement_noise: float = 1.0,
                 max_age: int = 5,
                 min_hits: int = 3,
                 association_threshold: float = 5.0):
        """
        Initialize Kalman tracker
        
        Args:
            process_noise: Process noise covariance scaling
            measurement_noise: Measurement noise covariance scaling
            max_age: Maximum number of frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            association_threshold: Maximum Mahalanobis distance for data association
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_age = max_age
        self.min_hits = min_hits
        self.association_threshold = association_threshold
        
        self.tracks: List[TrackedObject] = []
        self.next_track_id = 0
        self.dt = 0.1  # Default time step in seconds
        
    def set_dt(self, dt: float):
        """Set time step for prediction"""
        self.dt = dt
        
    def update(self, detections: List[RadarDetection]) -> List[TrackedObject]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of current frame detections
            
        Returns:
            List of confirmed tracks
        """
        # Predict existing tracks
        for track in self.tracks:
            self._predict(track)
        
        # Associate detections with tracks
        matched_tracks, matched_detections, unmatched_tracks, unmatched_detections = \
            self._associate(detections)
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_detections):
            self._update_track(self.tracks[track_idx], detections[det_idx])
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].misses += 1
            self.tracks[track_idx].confidence *= 0.9
        
        # Initialize new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._initialize_track(detections[det_idx])
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses < self.max_age]
        
        # Return confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.hits >= self.min_hits]
        return confirmed_tracks
    
    def _predict(self, track: TrackedObject):
        """Predict track state using constant velocity model"""
        # State transition matrix (constant velocity model)
        F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise covariance
        Q = np.eye(6) * self.process_noise
        Q[3:6, 3:6] *= 0.1  # Lower noise for velocity
        
        # Predict
        track.state = F @ track.state
        track.covariance = F @ track.covariance @ F.T + Q
        track.age += 1
    
    def _update_track(self, track: TrackedObject, detection: RadarDetection):
        """Update track with detection"""
        # Measurement matrix (we observe position and velocity)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement noise covariance
        R = np.eye(6) * self.measurement_noise
        
        # Measurement
        z = np.concatenate([detection.centroid, detection.velocity])
        
        # Innovation
        y = z - H @ track.state
        
        # Innovation covariance
        S = H @ track.covariance @ H.T + R
        
        # Kalman gain
        K = track.covariance @ H.T @ np.linalg.inv(S)
        
        # Update state
        track.state = track.state + K @ y
        
        # Update covariance
        track.covariance = (np.eye(6) - K @ H) @ track.covariance
        
        # Update track metadata
        track.hits += 1
        track.misses = 0
        track.confidence = min(1.0, track.confidence + 0.1)
        track.detections.append(detection)
    
    def _associate(self, detections: List[RadarDetection]):
        """
        Associate detections with tracks using Mahalanobis distance
        
        Returns:
            matched_tracks, matched_detections, unmatched_tracks, unmatched_detections
        """
        if len(self.tracks) == 0:
            return [], [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks))), []
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                # Mahalanobis distance
                z = np.concatenate([detection.centroid, detection.velocity])
                H = np.eye(6)
                y = z - H @ track.state
                S = H @ track.covariance @ H.T + np.eye(6) * self.measurement_noise
                
                try:
                    dist = np.sqrt(y.T @ np.linalg.inv(S) @ y)
                    cost_matrix[i, j] = dist
                except np.linalg.LinAlgError:
                    cost_matrix[i, j] = 1e10
        
        # Hungarian algorithm (simplified greedy matching)
        matched_tracks = []
        matched_detections = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        
        # Greedy matching
        while len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
            # Find minimum cost
            min_cost = 1e10
            min_i = -1
            min_j = -1
            
            for i in unmatched_tracks:
                for j in unmatched_detections:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i = i
                        min_j = j
            
            # Check if match is valid
            if min_cost < self.association_threshold:
                matched_tracks.append(min_i)
                matched_detections.append(min_j)
                unmatched_tracks.remove(min_i)
                unmatched_detections.remove(min_j)
            else:
                break
        
        return matched_tracks, matched_detections, unmatched_tracks, unmatched_detections
    
    def _initialize_track(self, detection: RadarDetection):
        """Initialize new track from detection"""
        # Initial state [x, y, z, vx, vy, vz]
        state = np.concatenate([detection.centroid, detection.velocity])
        
        # Initial covariance (high uncertainty)
        covariance = np.eye(6) * 10.0
        
        track = TrackedObject(
            track_id=self.next_track_id,
            state=state,
            covariance=covariance,
            detections=[detection],
            age=1,
            hits=1,
            misses=0,
            confidence=detection.confidence
        )
        
        self.tracks.append(track)
        self.next_track_id += 1
