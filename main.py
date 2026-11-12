#!/usr/bin/env python3
"""
VisionTrack: Automated Multi-Object Detection, Homography-Calibrated Tracking,
and Kinematic Trajectory Extraction Engine.

Entry point and CLI harness.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from visiontrack.detector import ObjectDetectorTracker
from visiontrack.exporter import DataExporter
from visiontrack.homography import HomographyCalibrator
from visiontrack.tracker import KinematicTracker
from visiontrack.visualizer import HeatmapAccumulator, Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visiontrack.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visiontrack",
        description="VisionTrack: Homography-Calibrated Multi-Object Detection & Kinematic Tracking Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Path to input video file (e.g., traffic.mp4) or camera index (e.g., 0).",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch interactive 4-point ROI ground calibration tool on the first frame.",
    )
    parser.add_argument(
        "--calibration-file",
        type=str,
        default="calibration.json",
        help="Path to homography calibration JSON file (load or save).",
    )
    parser.add_argument(
        "--roi-length",
        type=float,
        default=30.0,
        help="Ground plane ROI length in meters (longitudinal direction).",
    )
    parser.add_argument(
        "--roi-width",
        type=float,
        default=7.0,
        help="Ground plane ROI width in meters (transverse lane width).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics YOLO model weight file or name (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold for ByteTrack.",
    )
    parser.add_argument(
        "--tracker-config",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config file (bytetrack.yaml or botsort.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save trajectories.csv, summary.json, and heatmap.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run in headless mode without displaying OpenCV GUI window.",
    )
    parser.add_argument(
        "--no-kalman",
        action="store_true",
        help="Disable metric ground-plane Kalman filter state estimation (reverts to Savitzky-Golay).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the annotated video to <output-dir>/annotated_output.mp4.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cpu', 'cuda', '0', etc. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )

    return parser.parse_args()


def resolve_video_source(source: str) -> Union[int, str]:
    """Converts numeric string to int for webcam, or validates file existence."""
    if source.isdigit():
        return int(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path.resolve()}")
    return str(path)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Open Video Stream
    try:
        video_src = resolve_video_source(args.video)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        logger.error("Failed to open video source: %s", args.video)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Opened video stream: %s (Resolution: %dx%d, FPS: %.2f, Total Frames: %d)",
        args.video,
        width,
        height,
        fps,
        total_frames if total_frames > 0 else -1,
    )

    # 2. Homography Calibration
    calibrator = HomographyCalibrator()
    calib_file = Path(args.calibration_file)

    if args.calibrate:
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            logger.error("Could not read initial frame from video for calibration.")
            cap.release()
            return 1

        logger.info("Opening interactive 4-point homography calibration window...")
        calib_success = calibrator.calibrate_interactively(
            frame=first_frame,
            length_m=args.roi_length,
            width_m=args.roi_width,
            output_file=calib_file,
        )
        if not calib_success:
            logger.warning("Calibration was not completed. Proceeding with uncalibrated mode.")
        # Rewind video capture back to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    elif calib_file.is_file():
        logger.info("Loading existing calibration from %s", calib_file)
        calibrator.load(calib_file)
    else:
        logger.warning(
            "No calibration file '%s' found. Use '--calibrate' to set real-world ground metrics.",
            calib_file,
        )

    # 3. Initialize Core Modules
    detector = ObjectDetectorTracker(
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        tracker_config=args.tracker_config,
        device=args.device,
    )
    tracker = KinematicTracker(calibrator=calibrator, use_kalman=not args.no_kalman)
    visualizer = Visualizer(calibrator=calibrator)
    heatmap = HeatmapAccumulator(frame_shape=(height, width))
    exporter = DataExporter()

    # 4. Optional Video Writer
    video_writer = None
    if args.save_video:
        video_out_path = output_dir / "annotated_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
        logger.info("Saving annotated video stream to %s", video_out_path)

    # 5. Main Processing Loop
    window_name = "VisionTrack - Live Tracking & Kinematics"
    if not args.no_show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            logger.warning("GUI display unavailable; switching to headless mode (--no-show).")
            args.no_show = True

    frame_id = 0
    start_time = time.time()
    last_frame_for_bg = None
    all_seen_ids = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("End of video stream reached.")
                break

            frame_id += 1
            timestamp_sec = (frame_id - 1) / fps
            last_frame_for_bg = frame

            # Run detection and tracking
            tracked_dets = detector.track(frame)
            for d in tracked_dets:
                all_seen_ids.add(d.track_id)

            # Compute kinematics
            kinematic_states = tracker.update(
                frame_id=frame_id,
                timestamp_sec=timestamp_sec,
                detections=[d.to_dict() for d in tracked_dets],
            )

            # Accumulate structured records
            exporter.log_frame_states(kinematic_states)

            # Accumulate spatial occupancy for heatmap
            heatmap.update(kinematic_states, tracks=tracker.tracks)

            # Calculate processing FPS
            elapsed = time.time() - start_time
            proc_fps = frame_id / elapsed if elapsed > 0 else fps

            # Render visual overlays
            annotated_frame = visualizer.render_frame(
                frame=frame,
                kinematic_states=kinematic_states,
                tracks=tracker.tracks,
                frame_id=frame_id,
                fps=proc_fps,
                total_unique_objects=len(all_seen_ids),
            )

            if video_writer is not None:
                video_writer.write(annotated_frame)

            if not args.no_show:
                cv2.imshow(window_name, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    logger.info("Graceful user termination requested (key press).")
                    break

            if args.max_frames and frame_id >= args.max_frames:
                logger.info("Reached maximum frame limit (%d frames).", args.max_frames)
                break

    except KeyboardInterrupt:
        logger.info("Process interrupted by user (Ctrl+C). Saving accumulated data...")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    # 6. Export Artifacts
    logger.info("Processing complete. Finalizing structured outputs in '%s' ...", output_dir)
    csv_path = exporter.export_csv(output_dir / "trajectories.csv")

    all_tracks = tracker.get_all_tracks()
    summary_path = exporter.export_summary(
        filepath=output_dir / "summary.json",
        tracks=all_tracks,
        total_frames=frame_id,
        fps=fps,
        video_source=str(args.video),
    )

    heatmap_path = output_dir / "heatmap.png"
    heatmap.generate_heatmap(output_path=heatmap_path, background_frame=last_frame_for_bg)

    # 7. Print Console Summary
    total_time = time.time() - start_time
    avg_fps = frame_id / total_time if total_time > 0 else 0.0
    print("\n" + "=" * 65)
    print("               VISIONTRACK EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_id}")
    print(f"Processing Wall Time   : {total_time:.2f} s ({avg_fps:.1f} FPS)")
    print(f"Unique Objects Tracked : {len(all_tracks)}")
    print(f"Trajectory Records CSV : {csv_path.resolve()}")
    print(f"Summary Analytics JSON : {summary_path.resolve()}")
    print(f"Spatial Density Heatmap: {heatmap_path.resolve()}")
    print("=" * 65 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
