"""
VisionTrack Metric Ground-Plane Kalman Filter Module.

Implements a 2D Constant Velocity (CV) Kalman Filter formulated in real-world
metric Cartesian coordinates (meters, meters/second). Fuses noisy detector
ground-plane projections with Newtonian kinematic motion models to produce
optimal, smoothed state estimates and uncertainty bounds.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("visiontrack.kalman")


class GroundMetricKalmanFilter:
    """
    2D Constant Velocity (CV) Kalman Filter in metric ground space.

    State Vector:
        x = [X, Y, v_x, v_y]^T
        - X, Y: Cartesian ground position in meters
        - v_x, v_y: Ground velocity components in m/s

    Measurement Vector:
        z = [X_meas, Y_meas]^T (from homography projection)
    """

    def __init__(
        self,
        init_x: float,
        init_y: float,
        init_vx: float = 0.0,
        init_vy: float = 0.0,
        std_process_accel: float = 1.5,
        std_measurement_pos: float = 0.35,
    ) -> None:
        """
        Args:
            init_x: Initial ground X coordinate in meters.
            init_y: Initial ground Y coordinate in meters.
            init_vx: Initial ground X velocity in m/s.
            init_vy: Initial ground Y velocity in m/s.
            std_process_accel: Process noise standard deviation representing
                               random vehicle/pedestrian acceleration (m/s^2).
            std_measurement_pos: Measurement noise standard deviation representing
                                 detector ground localization uncertainty (meters).
        """
        # State vector: [X, Y, v_x, v_y]^T
        self.x = np.array([init_x, init_y, init_vx, init_vy], dtype=np.float64)

        # State error covariance matrix P (4x4)
        # High confidence in initial position, lower confidence in initial velocity
        self.P = np.diag([
            std_measurement_pos ** 2,
            std_measurement_pos ** 2,
            4.0,  # (2.0 m/s initial std)
            4.0,
        ]).astype(np.float64)

        # Measurement matrix H (2x4): maps state [X, Y, v_x, v_y] to measurement [X, Y]
        self.H_meas = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance R (2x2)
        self.R = np.diag([
            std_measurement_pos ** 2,
            std_measurement_pos ** 2,
        ]).astype(np.float64)

        self.q_var = std_process_accel ** 2
        self.time_since_update: int = 0
        self.total_updates: int = 1

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        """Constructs state transition matrix F for time step dt."""
        return np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _get_process_noise(self, dt: float) -> np.ndarray:
        """
        Constructs Continuous White Noise Acceleration (CWNA) discrete process noise Q.
        Q = G * G^T * q_var
        """
        dt2 = dt ** 2
        dt3 = dt ** 3 / 2.0
        dt4 = dt ** 4 / 4.0

        Q = np.array([
            [dt4, 0.0, dt3, 0.0],
            [0.0, dt4, 0.0, dt3],
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2],
        ], dtype=np.float64) * self.q_var

        return Q

    def predict(self, dt: float) -> np.ndarray:
        """
        Prediction Step: Projects state and covariance forward by dt seconds.

        x_{k|k-1} = F * x_{k-1|k-1}
        P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q

        Args:
            dt: Time difference in seconds.

        Returns:
            Predicted state vector x_{k|k-1}.
        """
        if dt <= 0:
            dt = 1e-3

        F = self._get_transition_matrix(dt)
        Q = self._get_process_noise(dt)

        # State projection
        self.x = F @ self.x

        # Covariance projection
        self.P = F @ self.P @ F.T + Q
        self.time_since_update += 1

        return self.x

    def update(self, meas_x: float, meas_y: float) -> np.ndarray:
        """
        Measurement Update Step: Corrects state estimate with new ground measurement.

        y = z - H * x
        S = H * P * H^T + R
        K = P * H^T * S^{-1}
        x = x + K * y
        P = (I - K * H) * P

        Args:
            meas_x: Observed ground X coordinate in meters.
            meas_y: Observed ground Y coordinate in meters.

        Returns:
            Updated state vector x_{k|k}.
        """
        z = np.array([meas_x, meas_y], dtype=np.float64)

        # Measurement residual (innovation)
        y = z - (self.H_meas @ self.x)

        # Innovation covariance
        S = self.H_meas @ self.P @ self.H_meas.T + self.R

        # Optimal Kalman gain
        K = self.P @ self.H_meas.T @ np.linalg.inv(S)

        # Updated state
        self.x = self.x + (K @ y)

        # Joseph form update for numerical stability: P = (I - KH)P(I - KH)^T + KRK^T
        I = np.eye(4, dtype=np.float64)
        I_KH = I - (K @ self.H_meas)
        self.P = I_KH @ self.P @ I_KH.T + (K @ self.R @ K.T)

        self.time_since_update = 0
        self.total_updates += 1

        return self.x

    @property
    def position(self) -> Tuple[float, float]:
        """Returns filtered ground position (X, Y) in meters."""
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Returns filtered ground velocity (v_x, v_y) in m/s."""
        return float(self.x[2]), float(self.x[3])

    @property
    def speed_mps(self) -> float:
        """Returns scalar ground speed in m/s."""
        vx, vy = self.x[2], self.x[3]
        return float(math.hypot(vx, vy))

    @property
    def speed_kmh(self) -> float:
        """Returns scalar ground speed in km/h."""
        return self.speed_mps * 3.6

    @property
    def heading_deg(self) -> float:
        """Returns movement direction in degrees [0, 360)."""
        vx, vy = self.x[2], self.x[3]
        if math.hypot(vx, vy) < 0.1:  # Stationary threshold
            return 0.0
        rad = math.atan2(vy, vx)
        return float((math.degrees(rad) + 360.0) % 360.0)

    @property
    def velocity_uncertainty(self) -> float:
        """Returns estimated 1-sigma standard deviation of speed in m/s."""
        # Trace of velocity submatrix in P
        var_v = float(self.P[2, 2] + self.P[3, 3])
        return math.sqrt(max(0.0, var_v))
