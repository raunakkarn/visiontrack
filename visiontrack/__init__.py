"""
VisionTrack: Automated Multi-Object Detection, Homography-Calibrated Tracking,
and Kinematic Trajectory Extraction Engine.
"""

from visiontrack.homography import HomographyCalibrator
from visiontrack.detector import ObjectDetectorTracker, TrackedDetection
from visiontrack.tracker import KinematicTracker, KinematicState, TrackState
from visiontrack.kalman import GroundMetricKalmanFilter
from visiontrack.visualizer import Visualizer, HeatmapAccumulator
from visiontrack.exporter import DataExporter

__version__ = "1.0.0"
__all__ = [
    "HomographyCalibrator",
    "ObjectDetectorTracker",
    "TrackedDetection",
    "KinematicTracker",
    "KinematicState",
    "TrackState",
    "GroundMetricKalmanFilter",
    "Visualizer",
    "HeatmapAccumulator",
    "DataExporter",
]
