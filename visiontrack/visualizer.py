"""
VisionTrack Visual Rendering & Heatmap Analytics Module.

Handles bounding box rendering, telemetry HUD, dynamic fading motion trails,
real-time kinematic speed overlays, and spatial trajectory density heatmaps.
"""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from visiontrack.homography import HomographyCalibrator
from visiontrack.tracker import KinematicState, TrackState

logger = logging.getLogger("visiontrack.visualizer")


def generate_color(track_id: int) -> Tuple[int, int, int]:
    """
    Generates a deterministic, vibrant BGR color for a given track ID
    using the golden ratio on the HSV color wheel.
    """
    golden_ratio_conjugate = 0.618033988749895
    h = (track_id * golden_ratio_conjugate) % 1.0
    s = 0.85
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)


class HeatmapAccumulator:
    """
    Accumulates 2D spatial track occupancy over time and renders
    high-resolution smoothed density heatmaps.
    """

    def __init__(self, frame_shape: Tuple[int, int]) -> None:
        """
        Args:
            frame_shape: (height, width) of the video stream.
        """
        self.height, self.width = frame_shape[:2]
        self.density = np.zeros((self.height, self.width), dtype=np.float32)
        self.total_points_accumulated: int = 0

    def update(self, states: List[KinematicState], tracks: Optional[Dict[int, TrackState]] = None) -> None:
        """
        Accumulates spatial occupancy points and trajectory lines from active states.
        """
        for state in states:
            px, py = state.pixel_point
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < self.width and 0 <= iy < self.height:
                # Accumulate point intensity
                cv2.circle(self.density, (ix, iy), radius=4, color=1.0, thickness=-1)
                self.total_points_accumulated += 1

                # If previous point is available in track, connect with anti-aliased line
                if tracks and state.track_id in tracks:
                    track = tracks[state.track_id]
                    if len(track.history_pixel) >= 2:
                        prev_pt = track.history_pixel[-2]
                        p_ix, p_iy = int(round(prev_pt[0])), int(round(prev_pt[1]))
                        if 0 <= p_ix < self.width and 0 <= p_iy < self.height:
                            cv2.line(self.density, (p_ix, p_iy), (ix, iy), color=0.5, thickness=2, lineType=cv2.LINE_AA)

    def generate_heatmap(
        self,
        output_path: Union[str, Path],
        background_frame: Optional[np.ndarray] = None,
        cmap: str = "inferno",
        alpha: float = 0.6,
        blur_ksize: int = 41,
    ) -> bool:
        """
        Smoothes accumulated spatial density with a Gaussian filter and exports
        a publication-ready analytics heatmap via Matplotlib.

        Args:
            output_path: Target PNG file path.
            background_frame: Optional BGR frame for alpha blending background.
            cmap: Matplotlib colormap name ('inferno', 'turbo', 'jet', etc.).
            alpha: Transparency weight for heatmap over background frame.
            blur_ksize: Gaussian blur kernel size (must be odd).

        Returns:
            True if heatmap was generated and saved successfully.
        """
        if self.total_points_accumulated == 0:
            logger.warning("No trajectory points accumulated; skipping heatmap generation.")
            return False

        if blur_ksize % 2 == 0:
            blur_ksize += 1

        # Smooth raw density accumulator
        smoothed = cv2.GaussianBlur(self.density, (blur_ksize, blur_ksize), 0)
        max_val = smoothed.max()
        if max_val > 0:
            normalized = smoothed / max_val
        else:
            normalized = smoothed

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

        if background_frame is not None:
            bg_rgb = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
            ax.imshow(bg_rgb)
            # Mask very low values to keep background transparent
            masked_density = np.ma.masked_where(normalized < 0.05, normalized)
            im = ax.imshow(masked_density, cmap=cmap, alpha=alpha, interpolation="bilinear")
        else:
            im = ax.imshow(normalized, cmap=cmap, interpolation="bilinear")

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("Normalized Spatial Trajectory Density", fontsize=11, fontweight="bold")

        ax.set_title(
            f"VisionTrack - Spatial Trajectory & Density Heatmap ({self.total_points_accumulated} pts)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_axis_off()

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=180)
        plt.close(fig)

        logger.info("Saved spatial density heatmap to %s", out_path.resolve())
        return True


class Visualizer:
    """
    Renders tracking overlays, bounding boxes, dynamic fading motion trails,
    kinematic telemetry tags, and telemetry HUD on video frames.
    """

    def __init__(
        self,
        calibrator: Optional[HomographyCalibrator] = None,
        draw_roi: bool = True,
        draw_trails: bool = True,
        draw_hud: bool = True,
    ) -> None:
        self.calibrator = calibrator
        self.draw_roi = draw_roi
        self.draw_trails = draw_trails
        self.draw_hud = draw_hud

    def render_frame(
        self,
        frame: np.ndarray,
        kinematic_states: List[KinematicState],
        tracks: Dict[int, TrackState],
        frame_id: int,
        fps: float,
        total_unique_objects: int,
    ) -> np.ndarray:
        """
        Draws visual overlays on the frame.

        Args:
            frame: Input BGR video frame.
            kinematic_states: Active objects detected in the current frame.
            tracks: Historical track states for trail rendering.
            frame_id: Current sequential frame index.
            fps: Current processing frame rate.
            total_unique_objects: Total unique IDs encountered so far.

        Returns:
            Annotated BGR video frame.
        """
        out = frame.copy()

        # 1. Draw Homography Ground ROI Polygon if available
        if self.draw_roi and self.calibrator and self.calibrator.is_calibrated:
            self._draw_calibration_roi(out)

        # 2. Draw Dynamic Fading Motion Trails
        if self.draw_trails:
            self._draw_motion_trails(out, kinematic_states, tracks)

        # 3. Draw Bounding Boxes and Kinematic Speed Overlays
        for state in kinematic_states:
            color = generate_color(state.track_id)
            self._draw_bounding_box(out, state, color)

        # 4. Draw Telemetry HUD
        if self.draw_hud:
            self._draw_hud(out, frame_id, fps, len(kinematic_states), total_unique_objects)

        return out

    def _draw_calibration_roi(self, frame: np.ndarray) -> None:
        """Draws the calibrated 4-point ground plane polygon with subtle shading."""
        if self.calibrator is None or self.calibrator.roi_pixels is None:
            return

        roi_pts = np.asarray(self.calibrator.roi_pixels, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_pts], (20, 180, 20))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # Draw boundary
        cv2.polylines(frame, [roi_pts], isClosed=True, color=(0, 255, 128), thickness=2, lineType=cv2.LINE_AA)

        # Corner nodes with labels
        labels = ["TL", "TR", "BR", "BL"]
        for i, pt in enumerate(roi_pts):
            cv2.circle(frame, tuple(pt), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"{labels[i]}",
                (pt[0] - 12, pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_motion_trails(
        self,
        frame: np.ndarray,
        states: List[KinematicState],
        tracks: Dict[int, TrackState],
    ) -> None:
        """Renders dynamic fading motion trails behind tracked contact points."""
        trail_overlay = frame.copy()

        for state in states:
            if state.track_id not in tracks:
                continue

            track = tracks[state.track_id]
            points = list(track.history_pixel)
            n_pts = len(points)
            if n_pts < 2:
                continue

            base_color = generate_color(state.track_id)

            for i in range(1, n_pts):
                pt1 = (int(round(points[i - 1][0])), int(round(points[i - 1][1])))
                pt2 = (int(round(points[i][0])), int(round(points[i][1])))

                # Calculate progress from oldest (0) to newest (1)
                factor = i / float(n_pts)
                thickness = max(1, int(round(1 + factor * 3)))

                # Modulate color brightness / fading
                segment_color = tuple(int(c * (0.3 + 0.7 * factor)) for c in base_color)
                cv2.line(trail_overlay, pt1, pt2, segment_color, thickness, cv2.LINE_AA)

        # Blend fading trails onto frame
        cv2.addWeighted(trail_overlay, 0.85, frame, 0.15, 0, frame)

    def _draw_bounding_box(
        self,
        frame: np.ndarray,
        state: KinematicState,
        color: Tuple[int, int, int],
    ) -> None:
        """Renders stylized bounding box, ground foot point, and kinematic tag."""
        x1, y1, x2, y2 = map(int, map(round, state.bbox))
        px, py = int(round(state.pixel_point[0])), int(round(state.pixel_point[1]))

        # Bounding box border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Foot contact point indicator
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 6, color, 1, cv2.LINE_AA)

        # Format kinematic tag: e.g. "ID: 4 | Car | 34.2 km/h"
        tag_text = f"ID: {state.track_id} | {state.class_name.capitalize()} | {state.speed_kmh:.1f} km/h"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(tag_text, font, font_scale, font_thickness)

        tag_y1 = max(0, y1 - text_h - 10)
        tag_y2 = tag_y1 + text_h + 8
        tag_x1 = max(0, x1)
        tag_x2 = min(frame.shape[1], tag_x1 + text_w + 14)

        # Draw tag background badge
        badge_overlay = frame.copy()
        cv2.rectangle(badge_overlay, (tag_x1, tag_y1), (tag_x2, tag_y2), (15, 15, 15), -1)
        cv2.addWeighted(badge_overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, 1, cv2.LINE_AA)

        # Text with high-contrast color
        text_pos = (tag_x1 + 7, tag_y2 - 5)
        cv2.putText(frame, tag_text, text_pos, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_id: int,
        fps: float,
        active_count: int,
        total_unique: int,
    ) -> None:
        """Renders top telemetry heads-up display (HUD)."""
        h, w = frame.shape[:2]
        hud_h = 42
        hud_w = min(w, 540)

        # Dark glassmorphism banner
        hud_bg = frame.copy()
        cv2.rectangle(hud_bg, (10, 10), (hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(hud_bg, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (10, 10), (hud_w, 10 + hud_h), (60, 60, 60), 1, cv2.LINE_AA)

        # Status text
        calib_str = "CALIB: ON" if (self.calibrator and self.calibrator.is_calibrated) else "CALIB: OFF"
        hud_text = f"Frame: {frame_id:04d} | FPS: {fps:.1f} | Active: {active_count} | Total IDs: {total_unique} | {calib_str}"
        cv2.putText(
            frame,
            hud_text,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
