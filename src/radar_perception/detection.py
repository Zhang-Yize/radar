"""
Object detection module using CFAR (Constant False Alarm Rate) algorithm
"""
import numpy as np
from typing import List, Tuple
from .data_structures import RadarPoint, RadarDetection
from .signal_processing import apply_cfar_2d


class CFARDetector:
    """Implements CFAR-based object detection for radar"""
    
    def __init__(self, 
                 guard_cells: Tuple[int, int] = (2, 2),
                 training_cells: Tuple[int, int] = (4, 4),
                 pfa: float = 1e-6,
                 min_snr: float = 10.0,
                 clustering_threshold: float = 2.0):
        """
        Initialize CFAR detector
        
        Args:
            guard_cells: (range, doppler) guard cell sizes
            training_cells: (range, doppler) training cell sizes
            pfa: Probability of false alarm
            min_snr: Minimum SNR threshold in dB
            clustering_threshold: Maximum distance for clustering detections
        """
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.pfa = pfa
        self.min_snr = min_snr
        self.clustering_threshold = clustering_threshold
        
    def detect(self, 
               range_doppler_map: np.ndarray,
               range_bins: np.ndarray,
               velocity_bins: np.ndarray,
               azimuth_bins: np.ndarray = None,
               timestamp: float = 0.0) -> List[RadarDetection]:
        """
        Detect objects in range-Doppler map
        
        Args:
            range_doppler_map: 2D power map (doppler_bins x range_bins)
            range_bins: Array of range values in meters
            velocity_bins: Array of velocity values in m/s
            azimuth_bins: Optional array of azimuth values in radians
            timestamp: Time of measurement
            
        Returns:
            List of radar detections
        """
        # Apply CFAR detection
        detection_mask = apply_cfar_2d(
            range_doppler_map,
            self.guard_cells,
            self.training_cells,
            self.pfa
        )
        
        # Get detection coordinates
        det_doppler_idx, det_range_idx = np.where(detection_mask)
        
        if len(det_doppler_idx) == 0:
            return []
        
        # Create radar points
        points = []
        for d_idx, r_idx in zip(det_doppler_idx, det_range_idx):
            range_val = range_bins[r_idx]
            velocity_val = velocity_bins[d_idx]
            power = range_doppler_map[d_idx, r_idx]
            
            # Estimate SNR (simplified)
            noise_floor = np.median(range_doppler_map)
            snr = 10 * np.log10(power / (noise_floor + 1e-10))
            
            # Skip low SNR detections
            if snr < self.min_snr:
                continue
            
            # Default azimuth if not provided
            azimuth = 0.0 if azimuth_bins is None else azimuth_bins[0]
            
            point = RadarPoint(
                range=range_val,
                azimuth=azimuth,
                elevation=0.0,  # Assume 2D radar
                velocity=velocity_val,
                rcs=power,
                snr=snr
            )
            points.append(point)
        
        if not points:
            return []
        
        # Cluster points into detections
        detections = self._cluster_points(points, timestamp)
        
        return detections
    
    def _cluster_points(self, points: List[RadarPoint], timestamp: float) -> List[RadarDetection]:
        """
        Cluster radar points into detections using simple distance-based clustering
        
        Args:
            points: List of radar points
            timestamp: Time of measurement
            
        Returns:
            List of clustered detections
        """
        if not points:
            return []
        
        # Convert to Cartesian coordinates
        positions = np.array([p.to_cartesian() for p in points])
        
        clusters = []
        used = np.zeros(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            if used[i]:
                continue
            
            # Start new cluster
            cluster_points = [point]
            cluster_indices = [i]
            used[i] = True
            
            # Find nearby points
            pos_i = positions[i]
            distances = np.linalg.norm(positions - pos_i, axis=1)
            nearby = (distances < self.clustering_threshold) & (~used)
            
            for j in np.where(nearby)[0]:
                cluster_points.append(points[j])
                cluster_indices.append(j)
                used[j] = True
            
            clusters.append(cluster_points)
        
        # Create detections from clusters
        detections = []
        for cluster in clusters:
            # Calculate centroid
            centroids = np.array([p.to_cartesian() for p in cluster])
            centroid = np.mean(centroids, axis=0)
            
            # Calculate average velocity
            velocities = np.array([p.velocity for p in cluster])
            avg_velocity = np.mean(velocities)
            
            # Velocity direction based on centroid direction
            direction = centroid / (np.linalg.norm(centroid) + 1e-10)
            velocity_vec = direction * avg_velocity
            
            # Average RCS and SNR
            avg_rcs = np.mean([p.rcs for p in cluster])
            avg_snr = np.mean([p.snr for p in cluster])
            
            # Confidence based on cluster size and SNR
            confidence = min(1.0, len(cluster) / 10.0 * (avg_snr / 20.0))
            
            detection = RadarDetection(
                points=cluster,
                centroid=centroid,
                velocity=velocity_vec,
                rcs=avg_rcs,
                confidence=confidence,
                timestamp=timestamp
            )
            detections.append(detection)
        
        return detections
