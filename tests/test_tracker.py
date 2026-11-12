"""
Unit tests for KinematicTracker module.
"""

import numpy as np
import pytest

from visiontrack.homography import HomographyCalibrator
from visiontrack.tracker import KinematicTracker


@pytest.fixture
def calibrated_tracker():
    calibrator = HomographyCalibrator()
    # Simple rectangular mapping: 100 pixels = 10 meters (10 px/meter)
    roi_pixels = np.array(
        [
            [100.0, 100.0],
            [200.0, 100.0],
            [200.0, 400.0],
            [100.0, 400.0],
        ],
        dtype=np.float32,
    )
    # width = 10.0 m (100 px), length = 30.0 m (300 px)
    calibrator.compute_homography(roi_pixels, length_m=30.0, width_m=10.0)
    return KinematicTracker(calibrator=calibrator, savgol_window=5, max_stale_frames=5)


def test_foot_point_calculation():
    bbox = (100.0, 50.0, 200.0, 150.0)
    xc, ymax = KinematicTracker.compute_foot_point(bbox)
    assert xc == 150.0
    assert ymax == 150.0


def test_linear_kinematics(calibrated_tracker: KinematicTracker):
    """
    Test moving vehicle with known ground speed.
    Ground: moving along +Y from (5.0, 0.0) to (5.0, 20.0) over 2 seconds (10 m/s = 36 km/h).
    """
    fps = 10.0
    dt = 1.0 / fps
    total_seconds = 2.0
    n_frames = int(total_seconds * fps)

    # In pixels:
    # Top-Left is at (100, 100) -> (0, 0)
    # X=5.0m -> px=150.0
    # Y from 0.0m (py=100) to 20.0m (py=300)
    states = []
    for f in range(n_frames):
        t = f * dt
        y_meter = (20.0 / total_seconds) * t  # 10 m/s
        py = 100.0 + (y_meter / 30.0) * 300.0  # 10 px/m
        px = 150.0

        # Create bbox with bottom center at (px, py)
        # width = 40, height = 60
        bbox = (px - 20, py - 60, px + 20, py)

        det = [
            {
                "track_id": 1,
                "class_name": "car",
                "confidence": 0.9,
                "bbox": bbox,
            }
        ]

        active_states = calibrated_tracker.update(frame_id=f + 1, timestamp_sec=t, detections=det)
        assert len(active_states) == 1
        states.append(active_states[0])

    # Check kinematics at final frames (after smoothing settles)
    last_state = states[-1]
    # Expected speed ~ 10 m/s = 36 km/h
    assert pytest.approx(last_state.speed_mps, rel=0.15) == 10.0
    assert pytest.approx(last_state.speed_kmh, rel=0.15) == 36.0

    # Heading: moving along +Y axis -> atan2(dy, dx) with dy > 0, dx = 0 is 90 degrees
    assert pytest.approx(last_state.heading_deg, abs=2.0) == 90.0

    # Total distance traversed ~ 19.0 m (initial step was at t=0)
    assert pytest.approx(last_state.total_dist_m, rel=0.1) == 19.0


def test_stale_track_archiving(calibrated_tracker: KinematicTracker):
    # Detection in frame 1
    calibrated_tracker.update(
        frame_id=1,
        timestamp_sec=0.0,
        detections=[{"track_id": 99, "class_name": "pedestrian", "confidence": 0.8, "bbox": (10, 10, 30, 50)}],
    )
    assert 99 in calibrated_tracker.tracks

    # Frames 2..7 without track 99 (exceeds max_stale_frames = 5)
    for f in range(2, 8):
        calibrated_tracker.update(frame_id=f, timestamp_sec=f * 0.1, detections=[])

    assert 99 not in calibrated_tracker.tracks
    assert 99 in calibrated_tracker.archived_tracks
