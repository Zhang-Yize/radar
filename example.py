"""
Example usage of the radar perception algorithm
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from radar_perception import (
    RangeProcessor,
    DopplerProcessor,
    CFARDetector,
    KalmanTracker
)


def simulate_radar_frame():
    """Simulate a radar frame with synthetic data"""
    # Radar parameters
    num_samples = 256
    num_chirps = 128
    sample_rate = 5e6  # 5 MHz
    chirp_bandwidth = 100e6  # 100 MHz
    chirp_period = 50e-6  # 50 microseconds
    wavelength = 0.004  # 4mm (77 GHz radar)
    
    # Create synthetic ADC data with a target
    adc_data = np.random.randn(num_chirps, num_samples) * 0.1 + \
               1j * np.random.randn(num_chirps, num_samples) * 0.1
    
    # Add a simulated target at range bin 64, doppler bin 80
    target_range_bin = 64
    target_doppler_bin = 80
    target_amplitude = 50.0  # Increased amplitude for better detection
    
    for chirp in range(num_chirps):
        phase = 2 * np.pi * target_doppler_bin * chirp / num_chirps
        adc_data[chirp, target_range_bin] += target_amplitude * np.exp(1j * phase)
    
    return adc_data, num_samples, num_chirps, sample_rate, chirp_bandwidth, chirp_period, wavelength


def main():
    """Main example demonstrating radar perception pipeline"""
    print("=" * 60)
    print("Radar Perception Algorithm Example")
    print("=" * 60)
    
    # Initialize processors
    print("\n1. Initializing signal processors...")
    adc_data, num_samples, num_chirps, sample_rate, chirp_bandwidth, chirp_period, wavelength = simulate_radar_frame()
    
    range_processor = RangeProcessor(
        num_samples=num_samples,
        sample_rate=sample_rate,
        chirp_bandwidth=chirp_bandwidth
    )
    
    doppler_processor = DopplerProcessor(
        num_chirps=num_chirps,
        chirp_period=chirp_period,
        wavelength=wavelength
    )
    
    print(f"   Range resolution: {range_processor.range_resolution:.2f} m")
    print(f"   Velocity resolution: {doppler_processor.velocity_resolution:.2f} m/s")
    
    # Process signal
    print("\n2. Processing radar signals...")
    range_profile = range_processor.process(adc_data)
    print(f"   Range profile shape: {range_profile.shape}")
    
    range_doppler_map = doppler_processor.process(range_profile)
    print(f"   Range-Doppler map shape: {range_doppler_map.shape}")
    print(f"   Max power in map: {np.max(range_doppler_map):.2f}")
    
    # Detect objects
    print("\n3. Detecting objects with CFAR...")
    detector = CFARDetector(
        guard_cells=(2, 2),
        training_cells=(4, 4),
        pfa=1e-5,  # Balanced false alarm rate
        min_snr=12.0,  # Higher threshold to reduce false positives
        clustering_threshold=3.0
    )
    
    range_bins = range_processor.get_range_bins()
    velocity_bins = doppler_processor.get_velocity_bins()
    
    detections = detector.detect(
        range_doppler_map,
        range_bins,
        velocity_bins,
        timestamp=0.0
    )
    
    print(f"   Number of detections: {len(detections)}")
    for i, det in enumerate(detections[:5]):  # Show only first 5 for brevity
        print(f"   Detection {i+1}:")
        print(f"     Position: ({det.centroid[0]:.2f}, {det.centroid[1]:.2f}, {det.centroid[2]:.2f}) m")
        print(f"     Velocity: ({det.velocity[0]:.2f}, {det.velocity[1]:.2f}, {det.velocity[2]:.2f}) m/s")
        print(f"     Confidence: {det.confidence:.2f}")
        print(f"     Number of points: {len(det.points)}")
    if len(detections) > 5:
        print(f"   ... and {len(detections) - 5} more detections")
    
    # Track objects
    print("\n4. Tracking objects with Kalman filter...")
    tracker = KalmanTracker(
        process_noise=1.0,
        measurement_noise=1.0,
        max_age=5,
        min_hits=3,
        association_threshold=5.0
    )
    tracker.set_dt(0.1)
    
    # Simulate multiple frames
    num_frames = 5
    for frame in range(num_frames):
        print(f"\n   Frame {frame + 1}:")
        
        # Generate new detections (in practice, process new radar data)
        # For demo, reuse detections with slight noise
        if detections:
            noisy_detections = []
            for det in detections:
                from radar_perception.data_structures import RadarDetection
                noisy_det = RadarDetection(
                    points=det.points,
                    centroid=det.centroid + np.random.randn(3) * 0.1,
                    velocity=det.velocity + np.random.randn(3) * 0.05,
                    rcs=det.rcs,
                    confidence=det.confidence,
                    timestamp=frame * 0.1
                )
                noisy_detections.append(noisy_det)
            
            tracks = tracker.update(noisy_detections)
            
            print(f"     Active tracks: {len(tracks)}")
            for track in tracks[:3]:  # Show only first 3 for brevity
                pos = track.get_position()
                vel = track.get_velocity()
                print(f"     Track ID {track.track_id}:")
                print(f"       Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
                print(f"       Velocity: ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) m/s")
                print(f"       Confidence: {track.confidence:.2f}")
                print(f"       Hits: {track.hits}, Age: {track.age}")
            if len(tracks) > 3:
                print(f"     ... and {len(tracks) - 3} more tracks")
        else:
            print("     No detections to track")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
