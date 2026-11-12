"""
Unit tests for DataExporter module.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from visiontrack.exporter import DataExporter, TRAJECTORY_CSV_COLUMNS
from visiontrack.tracker import KinematicState, TrackState


def test_schema_and_export(tmp_path: Path):
    exporter = DataExporter()

    mock_state = KinematicState(
        track_id=1,
        class_name="car",
        confidence=0.88,
        bbox=(120.0, 80.0, 180.0, 160.0),
        pixel_point=(150.0, 160.0),
        ground_point=(3.5, 12.0),
        timestamp_sec=1.5,
        frame_id=45,
        speed_mps=11.1,
        speed_kmh=40.0,
        accel_mps2=0.5,
        heading_deg=89.5,
        total_dist_m=15.2,
    )

    exporter.log_frame_states([mock_state])
    df = exporter.to_dataframe()

    # Verify column count and exact schema
    assert list(df.columns) == TRAJECTORY_CSV_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["track_id"] == 1
    assert df.iloc[0]["class_name"] == "car"
    assert df.iloc[0]["speed_kmh"] == 40.0

    csv_path = tmp_path / "trajectories.csv"
    exporter.export_csv(csv_path)
    assert csv_path.exists()

    # Read back and verify
    df_read = pd.read_csv(csv_path)
    assert list(df_read.columns) == TRAJECTORY_CSV_COLUMNS
    assert len(df_read) == 1


def test_summary_json(tmp_path: Path):
    exporter = DataExporter()

    mock_track = TrackState(
        track_id=1,
        class_name="car",
        first_frame=1,
        first_timestamp=0.0,
        last_frame=30,
        last_timestamp=1.0,
        total_distance_m=12.5,
        peak_speed_kmh=45.0,
        total_frames_seen=30,
    )
    mock_track.history_smoothed_speeds.append(10.0)

    summary_file = tmp_path / "summary.json"
    exporter.export_summary(
        filepath=summary_file,
        tracks={1: mock_track},
        total_frames=30,
        fps=30.0,
        video_source="test.mp4",
    )

    assert summary_file.exists()
    with open(summary_file, "r") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "class_distribution" in data
    assert "kinematics_summary" in data
    assert "tracks" in data
    assert data["class_distribution"]["car"] == 1
    assert data["metadata"]["total_frames_processed"] == 30
