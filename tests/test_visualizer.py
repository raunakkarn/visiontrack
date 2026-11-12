"""
Unit tests for Visualizer and HeatmapAccumulator modules.
"""

from pathlib import Path
import cv2
import numpy as np

from visiontrack.homography import HomographyCalibrator
from visiontrack.tracker import KinematicState, TrackState
from visiontrack.visualizer import HeatmapAccumulator, Visualizer


def test_heatmap_generation(tmp_path: Path):
    h, w = 480, 640
    heatmap = HeatmapAccumulator(frame_shape=(h, w))

    # Mock states with a moving object
    states = [
        KinematicState(
            track_id=1,
            class_name="car",
            confidence=0.9,
            bbox=(100, 100, 160, 180),
            pixel_point=(130.0 + i * 5, 180.0 + i * 5),
            ground_point=(2.0, 5.0 + i * 0.5),
            timestamp_sec=i * 0.1,
            frame_id=i + 1,
            speed_mps=5.0,
            speed_kmh=18.0,
            accel_mps2=0.0,
            heading_deg=45.0,
            total_dist_m=i * 0.5,
        )
        for i in range(20)
    ]

    heatmap.update(states)
    assert heatmap.total_points_accumulated == 20

    out_file = tmp_path / "test_heatmap.png"
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    success = heatmap.generate_heatmap(output_path=out_file, background_frame=bg)

    assert success is True
    assert out_file.exists()
    assert out_file.stat().st_size > 1000


def test_visualizer_render():
    calibrator = HomographyCalibrator()
    roi_pixels = np.array([[50, 50], [250, 50], [250, 250], [50, 250]], dtype=np.float32)
    calibrator.compute_homography(roi_pixels, length_m=20.0, width_m=5.0)

    vis = Visualizer(calibrator=calibrator)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    state = KinematicState(
        track_id=7,
        class_name="truck",
        confidence=0.95,
        bbox=(100.0, 100.0, 200.0, 250.0),
        pixel_point=(150.0, 250.0),
        ground_point=(2.5, 15.0),
        timestamp_sec=1.0,
        frame_id=30,
        speed_mps=12.5,
        speed_kmh=45.0,
        accel_mps2=0.2,
        heading_deg=90.0,
        total_dist_m=12.5,
    )

    track = TrackState(
        track_id=7,
        class_name="truck",
        first_frame=1,
        first_timestamp=0.0,
        last_frame=30,
        last_timestamp=1.0,
    )
    track.history_pixel.append((150.0, 240.0))
    track.history_pixel.append((150.0, 250.0))

    annotated = vis.render_frame(
        frame=frame,
        kinematic_states=[state],
        tracks={7: track},
        frame_id=30,
        fps=30.0,
        total_unique_objects=1,
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    # Frame should have non-zero pixels from bounding box, trail, HUD, etc.
    assert np.count_nonzero(annotated) > 0
