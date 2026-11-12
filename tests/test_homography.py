"""
Unit tests for HomographyCalibrator module.
"""

from pathlib import Path
import numpy as np
import pytest

from visiontrack.homography import HomographyCalibrator


def test_homography_computation():
    calibrator = HomographyCalibrator()

    # Image coordinates: a trapezoid representing a perspective road lane
    # Top-Left, Top-Right, Bottom-Right, Bottom-Left
    roi_pixels = np.array(
        [
            [400.0, 300.0],  # TL
            [600.0, 300.0],  # TR
            [800.0, 700.0],  # BR
            [200.0, 700.0],  # BL
        ],
        dtype=np.float32,
    )
    length_m = 30.0  # meters
    width_m = 7.0    # meters

    H = calibrator.compute_homography(roi_pixels, length_m=length_m, width_m=width_m)
    assert calibrator.is_calibrated is True
    assert H.shape == (3, 3)

    # Validate mapping of corner points
    tl_g = calibrator.pixel_to_ground(400.0, 300.0)
    tr_g = calibrator.pixel_to_ground(600.0, 300.0)
    br_g = calibrator.pixel_to_ground(800.0, 700.0)
    bl_g = calibrator.pixel_to_ground(200.0, 700.0)

    np.testing.assert_allclose(tl_g, [0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(tr_g, [7.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(br_g, [7.0, 30.0], atol=1e-3)
    np.testing.assert_allclose(bl_g, [0.0, 30.0], atol=1e-3)


def test_inverse_mapping():
    calibrator = HomographyCalibrator()
    roi_pixels = np.array(
        [
            [100.0, 100.0],
            [500.0, 100.0],
            [600.0, 500.0],
            [50.0, 500.0],
        ],
        dtype=np.float32,
    )
    calibrator.compute_homography(roi_pixels, length_m=20.0, width_m=5.0)

    # Map ground to pixel
    px, py = calibrator.ground_to_pixel(5.0, 20.0)
    np.testing.assert_allclose([px, py], [600.0, 500.0], atol=1e-2)

    # Batch test
    batch_pts = np.array([[100.0, 100.0], [500.0, 100.0]], dtype=np.float32)
    ground_batch = calibrator.pixel_to_ground_batch(batch_pts)
    assert ground_batch.shape == (2, 2)
    np.testing.assert_allclose(ground_batch[0], [0.0, 0.0], atol=1e-3)
    np.testing.assert_allclose(ground_batch[1], [5.0, 0.0], atol=1e-3)


def test_serialization(tmp_path: Path):
    calibrator = HomographyCalibrator()
    roi_pixels = np.array(
        [[200.0, 200.0], [400.0, 200.0], [500.0, 600.0], [100.0, 600.0]],
        dtype=np.float32,
    )
    calibrator.compute_homography(roi_pixels, length_m=25.0, width_m=6.0)

    save_path = tmp_path / "calibration.json"
    calibrator.save(save_path)
    assert save_path.exists()

    calibrator2 = HomographyCalibrator(calibration_file=save_path)
    assert calibrator2.is_calibrated is True
    assert calibrator2.width_m == 6.0
    assert calibrator2.length_m == 25.0
    np.testing.assert_allclose(calibrator.H, calibrator2.H, atol=1e-6)


def test_invalid_dimensions():
    calibrator = HomographyCalibrator()
    pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    with pytest.raises(ValueError, match="Real-world dimensions must be > 0"):
        calibrator.compute_homography(pts, length_m=-5.0, width_m=2.0)
