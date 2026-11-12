#!/usr/bin/env python3
"""
Synthetic Traffic Video Generator for VisionTrack.

Generates a realistic perspective roadway video with road markings, lane dividers,
and moving dynamic objects (cars, pedestrians) with known kinematics.
Also creates a matching calibration.json file.
"""

from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np


def generate_traffic_video(
    output_path: str = "sample_traffic.mp4",
    calibration_path: str = "calibration.json",
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    duration_sec: float = 6.0,
) -> None:
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Road perspective quad in image space
    tl_pt = (490, 240)
    tr_pt = (790, 240)
    br_pt = (1090, 680)
    bl_pt = (190, 680)
    road_poly = np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.int32)

    # Pre-generate calibration JSON for this scene
    # Length = 45.0m, Width = 8.0m
    real_length_m = 45.0
    real_width_m = 8.0

    src_pts = np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.float32)
    dst_pts = np.array(
        [[0.0, 0.0], [real_width_m, 0.0], [real_width_m, real_length_m], [0.0, real_length_m]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    calib_dict = {
        "roi_pixels": src_pts.tolist(),
        "real_world_width_m": real_width_m,
        "real_world_length_m": real_length_m,
        "homography_matrix": H.tolist(),
        "inverse_homography_matrix": H_inv.tolist(),
        "ground_polygon_m": dst_pts.tolist(),
    }
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calib_dict, f, indent=2)

    print(f"Generated calibration configuration: {calibration_path}")

    # Objects in the scene:
    # 1. Car A (Right lane, moving away or towards camera):
    #    Starts near y=260 (top), moves toward y=660 (bottom)
    # 2. Car B (Left lane, moving towards camera):
    #    Starts near y=280, moves to y=650
    # 3. Pedestrian on left sidewalk:
    #    Moves steadily along sidewalk

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        prog = frame_idx / total_frames

        # Create background canvas: sky + grass
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky
        frame[0:240, :] = (210, 180, 140)  # Light dusty blue/grey
        # Grass ground
        frame[240:height, :] = (45, 95, 45)  # Dark green grass

        # Sidewalks
        sidewalk_l = np.array([[430, 240], [490, 240], [190, 680], [100, 680]], dtype=np.int32)
        sidewalk_r = np.array([[790, 240], [850, 240], [1180, 680], [1090, 680]], dtype=np.int32)
        cv2.fillPoly(frame, [sidewalk_l], (170, 175, 175))
        cv2.fillPoly(frame, [sidewalk_r], (170, 175, 175))

        # Road asphalt
        cv2.fillPoly(frame, [road_poly], (50, 50, 50))
        cv2.polylines(frame, [road_poly], isClosed=True, color=(240, 240, 240), thickness=3)

        # Center dashed yellow line
        mid_top = (int((tl_pt[0] + tr_pt[0]) / 2), 240)
        mid_bot = (int((bl_pt[0] + br_pt[0]) / 2), 680)

        n_dashes = 12
        for d in range(n_dashes):
            if d % 2 == 0:
                s_frac = d / n_dashes
                e_frac = (d + 1) / n_dashes
                p1 = (
                    int(mid_top[0] + s_frac * (mid_bot[0] - mid_top[0])),
                    int(mid_top[1] + s_frac * (mid_bot[1] - mid_top[1])),
                )
                p2 = (
                    int(mid_top[0] + e_frac * (mid_bot[0] - mid_top[0])),
                    int(mid_top[1] + e_frac * (mid_bot[1] - mid_top[1])),
                )
                cv2.line(frame, p1, p2, (0, 215, 255), 3, cv2.LINE_AA)

        # Draw moving objects
        # 1. Car A (Right Lane)
        # Position progresses from near horizon to foreground
        car1_t = (prog * 1.3) % 1.0
        car1_y = int(270 + car1_t * 370)
        # Perspective scaling of car
        car1_scale = 0.35 + car1_t * 0.85
        w_car1 = int(75 * car1_scale)
        h_car1 = int(120 * car1_scale)
        # Interpolate x along right lane center
        right_lane_top_x = (tl_pt[0] + tr_pt[0]) / 2 + 80
        right_lane_bot_x = (bl_pt[0] + br_pt[0]) / 2 + 250
        car1_x = int(right_lane_top_x + car1_t * (right_lane_bot_x - right_lane_top_x))

        c1_x1, c1_y1 = car1_x - w_car1 // 2, car1_y - h_car1
        c1_x2, c1_y2 = car1_x + w_car1 // 2, car1_y

        # Draw vehicle body (blue sedan)
        cv2.rectangle(frame, (c1_x1, c1_y1), (c1_x2, c1_y2), (180, 50, 30), -1)
        # Windshield
        cv2.rectangle(frame, (c1_x1 + 6, c1_y1 + 10), (c1_x2 - 6, c1_y1 + int(h_car1 * 0.45)), (60, 60, 60), -1)
        # Headlights / tail lights
        cv2.circle(frame, (c1_x1 + 10, c1_y2 - 6), max(2, int(4 * car1_scale)), (0, 0, 220), -1)
        cv2.circle(frame, (c1_x2 - 10, c1_y2 - 6), max(2, int(4 * car1_scale)), (0, 0, 220), -1)

        # 2. Car B (Left Lane, silver car, slightly delayed)
        car2_t = ((prog + 0.35) * 1.1) % 1.0
        car2_y = int(270 + car2_t * 370)
        car2_scale = 0.35 + car2_t * 0.85
        w_car2 = int(80 * car2_scale)
        h_car2 = int(130 * car2_scale)

        left_lane_top_x = (tl_pt[0] + tr_pt[0]) / 2 - 80
        left_lane_bot_x = (bl_pt[0] + br_pt[0]) / 2 - 250
        car2_x = int(left_lane_top_x + car2_t * (left_lane_bot_x - left_lane_top_x))

        c2_x1, c2_y1 = car2_x - w_car2 // 2, car2_y - h_car2
        c2_x2, c2_y2 = car2_x + w_car2 // 2, car2_y

        cv2.rectangle(frame, (c2_x1, c2_y1), (c2_x2, c2_y2), (190, 190, 195), -1)
        cv2.rectangle(frame, (c2_x1 + 6, c2_y1 + 10), (c2_x2 - 6, c2_y1 + int(h_car2 * 0.45)), (40, 40, 40), -1)
        cv2.circle(frame, (c2_x1 + 10, c2_y2 - 6), max(2, int(4 * car2_scale)), (0, 0, 220), -1)
        cv2.circle(frame, (c2_x2 - 10, c2_y2 - 6), max(2, int(4 * car2_scale)), (0, 0, 220), -1)

        # 3. Pedestrian on left sidewalk
        ped_t = (prog * 0.5) % 1.0
        ped_y = int(320 + ped_t * 260)
        ped_scale = 0.4 + (ped_y - 320) / 260.0 * 0.5
        ped_x = int(350 - ped_t * 180)
        ped_h = int(60 * ped_scale)
        ped_w = int(24 * ped_scale)

        p_x1, p_y1 = ped_x - ped_w // 2, ped_y - ped_h
        p_x2, p_y2 = ped_x + ped_w // 2, ped_y

        # Pedestrian head
        head_radius = max(3, int(6 * ped_scale))
        cv2.circle(frame, (ped_x, p_y1 + head_radius), head_radius, (200, 180, 150), -1)
        # Torso
        cv2.rectangle(frame, (p_x1 + 2, p_y1 + head_radius * 2), (p_x2 - 2, p_y2 - int(ped_h * 0.4)), (20, 120, 220), -1)
        # Legs
        cv2.rectangle(frame, (p_x1 + 3, p_y2 - int(ped_h * 0.4)), (p_x2 - 3, p_y2), (30, 30, 90), -1)

        out.write(frame)

    out.release()
    print(f"Generated synthetic traffic video ({total_frames} frames, {width}x{height} @ {fps}fps): {output_path}")


if __name__ == "__main__":
    generate_traffic_video()
