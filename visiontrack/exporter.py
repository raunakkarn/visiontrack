"""
VisionTrack Data Logging & Export Module.

Streams per-frame kinematic trajectory data into a structured Pandas DataFrame,
exports high-resolution CSV datasets, and compiles comprehensive summary analytics JSON.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from visiontrack.tracker import KinematicState, TrackState

logger = logging.getLogger("visiontrack.exporter")

TRAJECTORY_CSV_COLUMNS = [
    "frame_id",
    "timestamp_sec",
    "track_id",
    "class_name",
    "confidence",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pixel_x",
    "pixel_y",
    "ground_x_m",
    "ground_y_m",
    "speed_kmh",
    "accel_ms2",
    "heading_deg",
    "total_dist_m",
]


class DataExporter:
    """
    Collects per-frame kinematic records, provides conversion to Pandas DataFrame,
    and handles structured CSV and JSON analytics export.
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def log_frame_states(self, states: List[KinematicState]) -> None:
        """Appends active kinematic states for the current frame."""
        for s in states:
            self.records.append(
                {
                    "frame_id": int(s.frame_id),
                    "timestamp_sec": round(float(s.timestamp_sec), 4),
                    "track_id": int(s.track_id),
                    "class_name": str(s.class_name),
                    "confidence": round(float(s.confidence), 4),
                    "bbox_x1": round(float(s.bbox[0]), 2),
                    "bbox_y1": round(float(s.bbox[1]), 2),
                    "bbox_x2": round(float(s.bbox[2]), 2),
                    "bbox_y2": round(float(s.bbox[3]), 2),
                    "pixel_x": round(float(s.pixel_point[0]), 2),
                    "pixel_y": round(float(s.pixel_point[1]), 2),
                    "ground_x_m": round(float(s.ground_point[0]), 3),
                    "ground_y_m": round(float(s.ground_point[1]), 3),
                    "speed_kmh": round(float(s.speed_kmh), 2),
                    "accel_ms2": round(float(s.accel_mps2), 3),
                    "heading_deg": round(float(s.heading_deg), 2),
                    "total_dist_m": round(float(s.total_dist_m), 3),
                }
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Converts accumulated records to a validated Pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=TRAJECTORY_CSV_COLUMNS)
        df = pd.DataFrame(self.records)
        return df[TRAJECTORY_CSV_COLUMNS]

    def export_csv(self, filepath: Union[str, Path]) -> Path:
        """
        Exports the dataset to CSV according to the required schema.

        Args:
            filepath: Destination CSV file path.

        Returns:
            Resolved Path object.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(out_path, index=False)
        logger.info("Exported %d trajectory records to %s", len(df), out_path.resolve())
        return out_path

    def export_summary(
        self,
        filepath: Union[str, Path],
        tracks: Dict[int, TrackState],
        total_frames: int,
        fps: float,
        video_source: str = "",
    ) -> Path:
        """
        Generates and saves a detailed summary analytics JSON file.

        Args:
            filepath: Destination JSON file path.
            tracks: Complete dictionary of active and archived TrackState instances.
            total_frames: Total video frames processed.
            fps: Video frames per second.
            video_source: Input video path or stream name.

        Returns:
            Resolved Path object.
        """
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        duration_sec = total_frames / fps if fps > 0 else 0.0

        # Class counts
        class_counts: Dict[str, int] = {}
        for t in tracks.values():
            class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1

        # Speeds per class
        peak_speed_by_class: Dict[str, float] = {}
        avg_speed_by_class: Dict[str, float] = {}

        if not df.empty:
            peak_overall = float(df["speed_kmh"].max())
            for cls_name, grp in df.groupby("class_name"):
                peak_speed_by_class[str(cls_name)] = round(float(grp["speed_kmh"].max()), 2)
                # Compute average speed only for non-zero/moving instances to avoid skew
                moving = grp[grp["speed_kmh"] > 0.5]
                avg_val = float(moving["speed_kmh"].mean()) if not moving.empty else float(grp["speed_kmh"].mean())
                avg_speed_by_class[str(cls_name)] = round(avg_val, 2)
        else:
            peak_overall = 0.0

        # Track durations and distances
        durations = []
        distances = []
        tracks_summary = []

        for tid, t in sorted(tracks.items(), key=lambda item: item[0]):
            dur = max(0.0, t.last_timestamp - t.first_timestamp)
            durations.append(dur)
            distances.append(t.total_distance_m)

            # Average speed for this track
            if len(t.history_smoothed_speeds) > 0:
                avg_spd = float(np.mean([s * 3.6 for s in t.history_smoothed_speeds]))
            else:
                avg_spd = 0.0

            tracks_summary.append(
                {
                    "track_id": tid,
                    "class_name": t.class_name,
                    "duration_sec": round(dur, 2),
                    "frames_seen": t.total_frames_seen,
                    "total_distance_m": round(t.total_distance_m, 2),
                    "average_speed_kmh": round(avg_spd, 2),
                    "peak_speed_kmh": round(t.peak_speed_kmh, 2),
                    "first_frame": t.first_frame,
                    "last_frame": t.last_frame,
                }
            )

        avg_duration = round(float(np.mean(durations)), 2) if durations else 0.0
        avg_distance = round(float(np.mean(distances)), 2) if distances else 0.0
        total_dist_all = round(float(np.sum(distances)), 2) if distances else 0.0

        summary = {
            "metadata": {
                "system": "VisionTrack Engine v1.0",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "video_source": video_source,
                "total_frames_processed": total_frames,
                "video_fps": round(fps, 2),
                "duration_seconds": round(duration_sec, 2),
                "total_unique_tracks": len(tracks),
                "total_trajectory_records": len(self.records),
            },
            "class_distribution": class_counts,
            "kinematics_summary": {
                "peak_speed_kmh_overall": round(peak_overall, 2),
                "peak_speed_by_class_kmh": peak_speed_by_class,
                "average_speed_by_class_kmh": avg_speed_by_class,
                "average_track_duration_sec": avg_duration,
                "average_track_distance_m": avg_distance,
                "total_distance_all_objects_m": total_dist_all,
            },
            "tracks": tracks_summary,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Saved summary analytics to %s", out_path.resolve())
        return out_path
