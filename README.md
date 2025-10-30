# Radar Perception Algorithm

A comprehensive radar perception system implementation for autonomous vehicles and robotics applications.

## Features

- **Signal Processing**: Range-Doppler processing for radar signals
  - Range FFT processing with windowing
  - Doppler FFT processing for velocity estimation
  - Configurable resolution and parameters

- **Object Detection**: CFAR (Constant False Alarm Rate) algorithm
  - 2D CFAR detection in range-Doppler domain
  - Adaptive thresholding for robust detection
  - Point clustering for multi-point targets

- **Object Tracking**: Kalman filter-based multi-object tracking
  - Constant velocity motion model
  - Mahalanobis distance-based data association
  - Track management with age and confidence scoring

- **Data Structures**: Clean, well-defined data structures
  - RadarPoint: Individual radar measurements
  - RadarDetection: Clustered detections
  - TrackedObject: Tracked objects over time

## Installation

```bash
# Clone the repository
git clone https://github.com/Zhang-Yize/radar.git
cd radar

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from radar_perception import (
    RangeProcessor,
    DopplerProcessor,
    CFARDetector,
    KalmanTracker
)
import numpy as np

# Initialize processors
range_processor = RangeProcessor(
    num_samples=256,
    sample_rate=5e6,
    chirp_bandwidth=100e6
)

doppler_processor = DopplerProcessor(
    num_chirps=128,
    chirp_period=50e-6,
    wavelength=0.004
)

# Process radar signals
range_profile = range_processor.process(adc_data)
range_doppler_map = doppler_processor.process(range_profile)

# Detect objects
detector = CFARDetector()
detections = detector.detect(
    range_doppler_map,
    range_processor.get_range_bins(),
    doppler_processor.get_velocity_bins()
)

# Track objects
tracker = KalmanTracker()
tracks = tracker.update(detections)
```

## Example

Run the included example to see the full pipeline in action:

```bash
python example.py
```

This demonstrates:
1. Signal processing (Range-Doppler processing)
2. Object detection with CFAR
3. Multi-frame tracking with Kalman filter

## Testing

Run the test suite to verify the installation:

```bash
# Run all tests
python tests/test_data_structures.py
python tests/test_signal_processing.py
python tests/test_detection.py
python tests/test_tracking.py
```

## Architecture

The radar perception pipeline consists of four main stages:

1. **Signal Processing**: Converts raw ADC samples to range-Doppler maps
   - Applies FFT along range and Doppler dimensions
   - Uses windowing to reduce sidelobes
   - Calculates power spectrum

2. **Detection**: Identifies potential targets using CFAR
   - Adaptive threshold based on local noise level
   - Filters detections by SNR
   - Clusters nearby points into detections

3. **Tracking**: Maintains tracks of detected objects over time
   - Predicts object states using Kalman filter
   - Associates detections with existing tracks
   - Manages track lifecycle (creation, update, deletion)

4. **Output**: Provides tracked objects with:
   - 3D position and velocity
   - Confidence scores
   - Detection history

## Requirements

- Python >= 3.8
- NumPy >= 1.21.0

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

This implementation is based on standard radar perception algorithms used in:
- Autonomous driving systems
- Robotics navigation
- Industrial automation
- Surveillance systems

Key algorithms implemented:
- Range-Doppler processing (FFT-based)
- CFAR detection (Cell Averaging CFAR)
- Kalman filtering for tracking
- Mahalanobis distance for data association