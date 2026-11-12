"""
Unit tests for GroundMetricKalmanFilter module.
"""

import math
import numpy as np
import pytest

from visiontrack.kalman import GroundMetricKalmanFilter


def test_kalman_initialization():
    kf = GroundMetricKalmanFilter(init_x=10.0, init_y=20.0, init_vx=0.0, init_vy=0.0)
    assert kf.position == (10.0, 20.0)
    assert kf.velocity == (0.0, 0.0)
    assert kf.speed_mps == 0.0
    assert kf.speed_kmh == 0.0
    assert kf.P.shape == (4, 4)


def test_kalman_predict():
    # Object at (5, 5) with velocity (2 m/s, 4 m/s)
    kf = GroundMetricKalmanFilter(init_x=5.0, init_y=5.0, init_vx=2.0, init_vy=4.0)

    # Predict 1.0 second forward
    dt = 1.0
    kf.predict(dt)

    pos_x, pos_y = kf.position
    # X_new = 5 + 2*1 = 7, Y_new = 5 + 4*1 = 9
    assert pytest.approx(pos_x, rel=1e-3) == 7.0
    assert pytest.approx(pos_y, rel=1e-3) == 9.0


def test_kalman_update_reduces_uncertainty():
    kf = GroundMetricKalmanFilter(init_x=0.0, init_y=0.0)
    kf.predict(dt=0.1)

    # Uncertainty before measurement update
    var_before = np.trace(kf.P)

    # Supply ground measurement
    kf.update(meas_x=0.2, meas_y=0.5)

    var_after = np.trace(kf.P)
    # Measurement update must contract the error covariance ellipsoid
    assert var_after < var_before


def test_kalman_noise_rejection():
    """
    Test linear vehicle moving at 10 m/s (36 km/h) along Y with additive measurement noise.
    Demonstrates that the Kalman Filter converges to ground truth and filters out jitter.
    """
    true_vx = 0.0
    true_vy = 10.0  # 10 m/s = 36 km/h
    dt = 0.1        # 10 FPS
    n_steps = 30

    np.random.seed(42)
    kf = GroundMetricKalmanFilter(init_x=0.0, init_y=0.0, std_process_accel=1.0, std_measurement_pos=0.3)

    raw_diff_speeds = []
    prev_meas_y = 0.0

    for i in range(1, n_steps + 1):
        true_y = i * true_vy * dt
        # Add random measurement noise
        noise = np.random.normal(0.0, 0.25)
        meas_y = true_y + noise
        meas_x = 0.0 + np.random.normal(0.0, 0.1)

        raw_diff_speed = (meas_y - prev_meas_y) / dt
        raw_diff_speeds.append(raw_diff_speed)
        prev_meas_y = meas_y

        kf.predict(dt)
        kf.update(meas_x, meas_y)

    # Kalman filtered speed at end of sequence
    final_kf_speed = kf.speed_mps

    # True speed is 10 m/s; Kalman filter should be within 10%
    assert pytest.approx(final_kf_speed, rel=0.10) == 10.0

    # Heading should be ~90 degrees (+Y axis)
    assert pytest.approx(kf.heading_deg, abs=3.0) == 90.0


def test_kalman_occlusion_coasting():
    """
    Test that when an object is occluded (no measurements), predict() continues
    projecting position along the velocity vector.
    """
    kf = GroundMetricKalmanFilter(init_x=0.0, init_y=0.0, init_vx=0.0, init_vy=5.0)

    # 3 frames with measurements at 5 m/s
    dt = 0.1
    for i in range(1, 4):
        kf.predict(dt)
        kf.update(0.0, i * 0.5)

    # 3 frames of occlusion (coast with predict only)
    for _ in range(3):
        kf.predict(dt)

    assert kf.time_since_update == 3
    # Speed is preserved during occlusion coasting
    assert pytest.approx(kf.speed_mps, rel=0.1) == 5.0
