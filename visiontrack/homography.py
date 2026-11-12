"""
VisionTrack Perspective Homography Calibration Module.

Provides interactive 4-point ground plane ROI selection, homography matrix
computation (cv2.getPerspectiveTransform), metric coordinate mapping (pixel <-> meters),
and serialization to/from JSON.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("visiontrack.homography")


@dataclass
class CalibrationData:
    """Container for homography calibration data."""

    roi_pixels: List[List[float]]
    real_world_width_m: float
    real_world_length_m: float
    homography_matrix: List[List[float]]
    inverse_homography_matrix: List[List[float]]
    ground_polygon_m: List[List[float]]


class HomographyCalibrator:
    """
    Manages perspective homography calibration between 2D image coordinates (pixels)
    and metric ground coordinates (meters).
    """

    def __init__(self, calibration_file: Optional[Union[str, Path]] = None) -> None:
        self.roi_pixels: Optional[np.ndarray] = None  # shape (4, 2)
        self.width_m: float = 0.0
        self.length_m: float = 0.0
        self.H: Optional[np.ndarray] = None  # 3x3 pixel -> ground
        self.H_inv: Optional[np.ndarray] = None  # 3x3 ground -> pixel
        self.is_calibrated: bool = False

        if calibration_file and Path(calibration_file).exists():
            self.load(calibration_file)

    @staticmethod
    def order_quad_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders 4 points into canonical order:
        [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
        """
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape != (4, 2):
            raise ValueError(f"Expected 4 points of shape (4, 2), got {pts.shape}")

        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left has the smallest sum, bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has the smallest difference (y - x), bottom-left largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def compute_homography(
        self,
        roi_pixels: Union[List[Tuple[float, float]], np.ndarray],
        length_m: float,
        width_m: float,
        auto_order: bool = False,
    ) -> np.ndarray:
        """
        Computes the 3x3 homography matrix mapping image pixels to ground meters.

        Order of points:
        1. Top-Left     -> (0.0, 0.0)
        2. Top-Right    -> (width_m, 0.0)
        3. Bottom-Right -> (width_m, length_m)
        4. Bottom-Left  -> (0.0, length_m)

        Args:
            roi_pixels: 4 image points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            length_m: Real-world longitudinal length of the ROI in meters.
            width_m: Real-world transverse width of the ROI in meters.
            auto_order: Whether to automatically sort points clockwise from top-left.

        Returns:
            The 3x3 homography matrix.
        """
        if length_m <= 0 or width_m <= 0:
            raise ValueError(f"Real-world dimensions must be > 0. Got L={length_m}, W={width_m}")

        src_pts = np.asarray(roi_pixels, dtype=np.float32)
        if src_pts.shape != (4, 2):
            raise ValueError(f"Expected (4, 2) shape for ROI points, got {src_pts.shape}")

        if auto_order:
            src_pts = self.order_quad_points(src_pts)

        # Target ground coordinates in meters
        dst_pts = np.array(
            [
                [0.0, 0.0],          # Top-Left
                [width_m, 0.0],      # Top-Right
                [width_m, length_m], # Bottom-Right
                [0.0, length_m],     # Bottom-Left
            ],
            dtype=np.float32,
        )

        try:
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        except cv2.error as e:
            raise ValueError(f"OpenCV failed to compute perspective transform: {e}") from e

        self.roi_pixels = src_pts
        self.width_m = float(width_m)
        self.length_m = float(length_m)
        self.H = H
        self.H_inv = H_inv
        self.is_calibrated = True

        logger.info(
            "Homography computed successfully (ROI L=%.2fm, W=%.2fm)",
            self.length_m,
            self.width_m,
        )
        return self.H

    def pixel_to_ground(self, px: float, py: float) -> Tuple[float, float]:
        """
        Transforms 2D pixel coordinates to ground plane coordinates (X, Y) in meters.

        Args:
            px: Pixel X coordinate (horizontal).
            py: Pixel Y coordinate (vertical).

        Returns:
            Tuple (X, Y) in meters.
        """
        if not self.is_calibrated or self.H is None:
            raise RuntimeError("HomographyCalibrator is not calibrated. Call compute_homography or load first.")

        pt = np.array([px, py, 1.0], dtype=np.float64)
        ground_pt = self.H @ pt
        w = ground_pt[2]
        if abs(w) < 1e-9:
            w = 1e-9 if w >= 0 else -1e-9
        return float(ground_pt[0] / w), float(ground_pt[1] / w)

    def pixel_to_ground_batch(self, points: np.ndarray) -> np.ndarray:
        """
        Batch transforms an array of pixel points (N, 2) to ground meters (N, 2).
        """
        if not self.is_calibrated or self.H is None:
            raise RuntimeError("HomographyCalibrator is not calibrated.")

        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, 2)
        pts_reshaped = pts.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts_reshaped, self.H)
        return transformed.reshape(-1, 2)

    def ground_to_pixel(self, gx: float, gy: float) -> Tuple[float, float]:
        """
        Transforms ground coordinates (X, Y) in meters back to pixel coordinates (px, py).
        """
        if not self.is_calibrated or self.H_inv is None:
            raise RuntimeError("HomographyCalibrator is not calibrated.")

        pt = np.array([gx, gy, 1.0], dtype=np.float64)
        px_pt = self.H_inv @ pt
        w = px_pt[2]
        if abs(w) < 1e-9:
            w = 1e-9 if w >= 0 else -1e-9
        return float(px_pt[0] / w), float(px_pt[1] / w)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes calibration data to a JSON file."""
        if not self.is_calibrated or self.H is None or self.roi_pixels is None:
            raise RuntimeError("Cannot save uncalibrated parameters.")

        dst_pts = [
            [0.0, 0.0],
            [self.width_m, 0.0],
            [self.width_m, self.length_m],
            [0.0, self.length_m],
        ]

        data = CalibrationData(
            roi_pixels=self.roi_pixels.tolist(),
            real_world_width_m=self.width_m,
            real_world_length_m=self.length_m,
            homography_matrix=self.H.tolist(),
            inverse_homography_matrix=self.H_inv.tolist(),
            ground_polygon_m=dst_pts,
        )

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(data), f, indent=2)

        logger.info("Saved calibration to %s", path.resolve())

    def load(self, filepath: Union[str, Path]) -> bool:
        """Loads calibration data from a JSON file."""
        path = Path(filepath)
        if not path.is_file():
            logger.warning("Calibration file does not exist: %s", path)
            return False

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.roi_pixels = np.array(raw["roi_pixels"], dtype=np.float32)
        self.width_m = float(raw["real_world_width_m"])
        self.length_m = float(raw["real_world_length_m"])
        self.H = np.array(raw["homography_matrix"], dtype=np.float64)
        if "inverse_homography_matrix" in raw:
            self.H_inv = np.array(raw["inverse_homography_matrix"], dtype=np.float64)
        else:
            self.H_inv = np.linalg.inv(self.H)
        self.is_calibrated = True

        logger.info("Loaded calibration from %s (L=%.2fm, W=%.2fm)", path, self.length_m, self.width_m)
        return True

    def calibrate_interactively(
        self,
        frame: np.ndarray,
        length_m: float = 30.0,
        width_m: float = 7.0,
        window_name: str = "VisionTrack - Homography Calibration",
        output_file: Optional[Union[str, Path]] = "calibration.json",
    ) -> bool:
        """
        Opens an interactive OpenCV window on the given frame to allow 4-point ROI selection.

        Controls:
          - Left Click: Place ROI corner point (TL -> TR -> BR -> BL)
          - 'r': Reset / clear selected points
          - 'c' or ENTER: Confirm selection and compute homography
          - 'q' or ESC: Cancel calibration

        Args:
            frame: Reference image frame (e.g. first frame of video stream).
            length_m: Ground plane length in meters.
            width_m: Ground plane width in meters.
            window_name: OpenCV window title.
            output_file: Path to automatically save calibration JSON on confirmation.

        Returns:
            True if calibration succeeded, False if cancelled or failed.
        """
        pts: List[Tuple[int, int]] = []
        clean_frame = frame.copy()
        h, w = clean_frame.shape[:2]

        # Check if GUI is supported / available
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            logger.error("Interactive calibration failed because GUI display is unavailable: %s", e)
            return False

        labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]

        def mouse_callback(event: int, x: int, y: int, flags: int, param: object) -> None:
            nonlocal pts
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(pts) < 4:
                    pts.append((x, y))
                    logger.debug("Selected point %d: (%d, %d)", len(pts), x, y)

        cv2.setMouseCallback(window_name, mouse_callback)

        confirmed = False
        while True:
            display = clean_frame.copy()

            # Draw HUD instructions
            hud_bg = display.copy()
            cv2.rectangle(hud_bg, (0, 0), (w, 80), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.75, display, 0.25, 0, display)

            cv2.putText(
                display,
                "VisionTrack Ground Calibration (Select 4 Rectangular Bounds)",
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Selected: {len(pts)}/4 | [r]: Reset | [c/Enter]: Confirm | [q/Esc]: Cancel | Dim: {width_m:.1f}m x {length_m:.1f}m",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw selected points and connecting polygon
            for i, pt in enumerate(pts):
                cv2.circle(display, pt, 7, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.circle(display, pt, 9, (255, 255, 255), 2, cv2.LINE_AA)
                lbl = labels[i] if i < len(labels) else f"{i+1}"
                cv2.putText(
                    display,
                    lbl,
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if len(pts) > 1:
                for i in range(len(pts) - 1):
                    cv2.line(display, pts[i], pts[i + 1], (0, 255, 0), 2, cv2.LINE_AA)

            if len(pts) == 4:
                # Close polygon
                cv2.line(display, pts[3], pts[0], (0, 255, 0), 2, cv2.LINE_AA)
                # Shaded overlay
                poly_overlay = display.copy()
                poly_pts = np.array(pts, dtype=np.int32)
                cv2.fillPoly(poly_overlay, [poly_pts], (0, 255, 0))
                cv2.addWeighted(poly_overlay, 0.25, display, 0.75, 0, display)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):  # 'q' or ESC
                logger.info("Interactive calibration cancelled by user.")
                break
            elif key == ord("r"):  # reset
                pts.clear()
                logger.info("Calibration points cleared.")
            elif key in (ord("c"), 10, 13, 32):  # 'c', Enter, Space
                if len(pts) == 4:
                    confirmed = True
                    break
                else:
                    logger.warning("Please select exactly 4 points before confirming.")

        cv2.destroyWindow(window_name)

        if confirmed and len(pts) == 4:
            self.compute_homography(pts, length_m=length_m, width_m=width_m, auto_order=False)
            if output_file:
                self.save(output_file)
            return True

        return False
