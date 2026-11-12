"""
VisionTrack Kinematic Tracking & Trajectory Analysis Module.

Maintains temporal history buffers, calculates bottom-center ground contact points,
and computes real-time kinematic parameters (instantaneous & smoothed velocity,
acceleration, heading, and cumulative distance) using metric ground plane coordinates.
"""

from __future__ import annotations

import collections
import logging
import math
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from visiontrack.homography import HomographyCalibrator
from visiontrack.kalman import GroundMetricKalmanFilter

logger = logging.getLogger("visiontrack.tracker")


@dataclass
class KinematicState:
    """Current kinematic state for a tracked object at a given frame."""

    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    pixel_point: Tuple[float, float]        # (px, py) ground foot point
    ground_point: Tuple[float, float]       # (X, Y) in meters
    timestamp_sec: float
    frame_id: int
    speed_mps: float
    speed_kmh: float
    accel_mps2: float
    heading_deg: float
    total_dist_m: float


@dataclass
class TrackState:
    """Historical state and kinematics accumulator for a single track ID."""

    track_id: int
    class_name: str
    first_frame: int
    first_timestamp: float
    last_frame: int
    last_timestamp: float
    last_confidence: float = 0.0

    # Historical buffers (fixed-size deques)
    history_pixel: Deque[Tuple[float, float]] = field(default_factory=lambda: collections.deque(maxlen=60))
    history_ground: Deque[Tuple[float, float]] = field(default_factory=lambda: collections.deque(maxlen=60))
    history_timestamps: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=60))
    history_raw_speeds: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=60))
    history_smoothed_speeds: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=60))

    # Real-time metrics
    current_speed_mps: float = 0.0
    current_speed_kmh: float = 0.0
    current_accel_mps2: float = 0.0
    current_heading_deg: float = 0.0
    total_distance_m: float = 0.0
    total_frames_seen: int = 0
    peak_speed_kmh: float = 0.0
    kalman_filter: Optional[GroundMetricKalmanFilter] = None


class KinematicTracker:
    """
    Tracks ground contact points and computes smoothed kinematic trajectories
    projected through perspective homography.
    """

    def __init__(
        self,
        calibrator: HomographyCalibrator,
        history_len: int = 60,
        savgol_window: int = 7,
        savgol_polyorder: int = 2,
        jitter_threshold_m: float = 0.05,
        max_stale_frames: int = 30,
        use_kalman: bool = True,
        kalman_process_std: float = 1.5,
        kalman_measure_std: float = 0.35,
    ) -> None:
        """
        Args:
            calibrator: Configured HomographyCalibrator instance.
            history_len: Max historical trajectory points to keep in memory per track.
            savgol_window: Window length for Savitzky-Golay filter (must be odd integer).
            savgol_polyorder: Polynomial order for Savitzky-Golay filter.
            jitter_threshold_m: Minimum displacement in meters below which motion is treated as sensor jitter.
            max_stale_frames: Remove tracks inactive for this many consecutive frames.
            use_kalman: Whether to enable 2D metric ground-plane Kalman state estimation.
            kalman_process_std: Process noise acceleration std in m/s^2.
            kalman_measure_std: Measurement noise position std in meters.
        """
        self.calibrator = calibrator
        self.history_len = history_len
        self.savgol_window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        self.savgol_polyorder = savgol_polyorder
        self.jitter_threshold_m = jitter_threshold_m
        self.max_stale_frames = max_stale_frames
        self.use_kalman = use_kalman
        self.kalman_process_std = kalman_process_std
        self.kalman_measure_std = kalman_measure_std

        self.tracks: Dict[int, TrackState] = {}
        self.archived_tracks: Dict[int, TrackState] = {}

    @staticmethod
    def compute_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Calculates bottom-center contact point (xc, ymax) of bounding box.
        bbox format: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        xc = (x1 + x2) / 2.0
        ymax = y2
        return float(xc), float(ymax)

    def _smooth_speeds(self, speeds: Deque[float]) -> float:
        """
        Applies Savitzky-Golay or rolling weighted window smoothing to recent speed history.
        """
        n = len(speeds)
        if n == 0:
            return 0.0
        if n < 3:
            return float(speeds[-1])
        if n < self.savgol_window:
            # Simple moving average for small sample sizes
            return float(np.mean(speeds))

        try:
            arr = np.array(speeds, dtype=np.float64)
            smoothed = savgol_filter(arr, window_length=self.savgol_window, polyorder=self.savgol_polyorder)
            return max(0.0, float(smoothed[-1]))
        except Exception:
            return max(0.0, float(np.mean(speeds)))

    def update(
        self,
        frame_id: int,
        timestamp_sec: float,
        detections: List[Dict],
    ) -> List[KinematicState]:
        """
        Updates track trajectories with current frame detections and computes kinematics.

        Args:
            frame_id: Sequential video frame index.
            timestamp_sec: Timestamp of current frame in seconds.
            detections: List of detection dicts containing keys:
                        'track_id', 'class_name', 'confidence', 'bbox'.

        Returns:
            List of KinematicState objects for active detections in this frame.
        """
        active_states: List[KinematicState] = []
        current_seen_ids = set()

        for det in detections:
            track_id = int(det["track_id"])
            class_name = str(det["class_name"])
            confidence = float(det["confidence"])
            bbox = tuple(map(float, det["bbox"]))
            current_seen_ids.add(track_id)

            # 1. Compute bottom-center foot point in pixels
            px, py = self.compute_foot_point(bbox)

            # 2. Map foot point to real-world ground coordinates (X, Y) in meters
            if self.calibrator.is_calibrated:
                try:
                    gx, gy = self.calibrator.pixel_to_ground(px, py)
                except Exception as e:
                    logger.debug("Failed to project point (%f, %f): %s", px, py, e)
                    gx, gy = 0.0, 0.0
            else:
                gx, gy = 0.0, 0.0

            # 3. Retrieve or initialize TrackState
            if track_id not in self.tracks:
                kf = (
                    GroundMetricKalmanFilter(
                        init_x=gx,
                        init_y=gy,
                        std_process_accel=self.kalman_process_std,
                        std_measurement_pos=self.kalman_measure_std,
                    )
                    if self.use_kalman
                    else None
                )
                track = TrackState(
                    track_id=track_id,
                    class_name=class_name,
                    first_frame=frame_id,
                    first_timestamp=timestamp_sec,
                    last_frame=frame_id,
                    last_timestamp=timestamp_sec,
                    last_confidence=confidence,
                    history_pixel=collections.deque(maxlen=self.history_len),
                    history_ground=collections.deque(maxlen=self.history_len),
                    history_timestamps=collections.deque(maxlen=self.history_len),
                    history_raw_speeds=collections.deque(maxlen=self.history_len),
                    history_smoothed_speeds=collections.deque(maxlen=self.history_len),
                    kalman_filter=kf,
                )
                self.tracks[track_id] = track
            else:
                track = self.tracks[track_id]
                track.class_name = class_name
                track.last_confidence = confidence

            # 4. Compute kinematic delta if previous state exists
            speed_mps = 0.0
            speed_kmh = 0.0
            accel_mps2 = 0.0
            heading_deg = track.current_heading_deg

            if len(track.history_ground) > 0 and len(track.history_timestamps) > 0:
                prev_gx, prev_gy = track.history_ground[-1]
                prev_t = track.history_timestamps[-1]
                dt = max(1e-4, timestamp_sec - prev_t)

                if self.use_kalman and track.kalman_filter is not None:
                    # Recursive Bayesian State Estimation (Predict -> Update)
                    track.kalman_filter.predict(dt)
                    track.kalman_filter.update(gx, gy)
                    gx, gy = track.kalman_filter.position
                    speed_mps = track.kalman_filter.speed_mps
                    speed_kmh = track.kalman_filter.speed_kmh

                    dx = gx - prev_gx
                    dy = gy - prev_gy
                    dist = math.hypot(dx, dy)

                    if dist < self.jitter_threshold_m:
                        dist = 0.0
                        speed_mps = 0.0
                        speed_kmh = 0.0
                    else:
                        heading_deg = track.kalman_filter.heading_deg

                    track.total_distance_m += dist
                    track.history_raw_speeds.append(speed_mps)
                else:
                    dx = gx - prev_gx
                    dy = gy - prev_gy
                    dist = math.hypot(dx, dy)

                    if dist < self.jitter_threshold_m:
                        dist = 0.0
                        raw_speed = 0.0
                    elif dt > 1e-4:
                        raw_speed = dist / dt
                        heading_rad = math.atan2(dy, dx)
                        heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0
                    else:
                        raw_speed = 0.0

                    track.total_distance_m += dist
                    track.history_raw_speeds.append(raw_speed)

                    # Smooth speed using moving average / Savitzky-Golay
                    smoothed_mps = self._smooth_speeds(track.history_raw_speeds)
                    speed_mps = smoothed_mps
                    speed_kmh = smoothed_mps * 3.6

                # Compute acceleration from filtered/smoothed speed change
                if len(track.history_smoothed_speeds) > 0 and dt > 1e-4:
                    prev_v = track.history_smoothed_speeds[-1]
                    accel_mps2 = (speed_mps - prev_v) / dt
                else:
                    accel_mps2 = 0.0
            else:
                track.history_raw_speeds.append(0.0)
                speed_mps = 0.0
                speed_kmh = 0.0
                accel_mps2 = 0.0

            # 5. Append to buffers and update real-time state
            track.history_pixel.append((px, py))
            track.history_ground.append((gx, gy))
            track.history_timestamps.append(timestamp_sec)
            track.history_smoothed_speeds.append(speed_mps)

            track.current_speed_mps = speed_mps
            track.current_speed_kmh = speed_kmh
            track.current_accel_mps2 = accel_mps2
            track.current_heading_deg = heading_deg
            track.last_frame = frame_id
            track.last_timestamp = timestamp_sec
            track.total_frames_seen += 1

            if speed_kmh > track.peak_speed_kmh:
                track.peak_speed_kmh = speed_kmh

            state = KinematicState(
                track_id=track_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                pixel_point=(px, py),
                ground_point=(gx, gy),
                timestamp_sec=timestamp_sec,
                frame_id=frame_id,
                speed_mps=speed_mps,
                speed_kmh=speed_kmh,
                accel_mps2=accel_mps2,
                heading_deg=heading_deg,
                total_dist_m=track.total_distance_m,
            )
            active_states.append(state)

        # 6. Archive stale tracks
        stale_ids = [
            tid
            for tid, trk in self.tracks.items()
            if (frame_id - trk.last_frame) > self.max_stale_frames and tid not in current_seen_ids
        ]
        for tid in stale_ids:
            self.archived_tracks[tid] = self.tracks.pop(tid)

        return active_states

    def get_all_tracks(self) -> Dict[int, TrackState]:
        """Returns union of all active and archived tracks."""
        combined = dict(self.archived_tracks)
        combined.update(self.tracks)
        return combined
