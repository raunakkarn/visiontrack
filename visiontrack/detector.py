"""
VisionTrack Detection & Multi-Object Association Engine.

Integrates Ultralytics YOLOv8 with ByteTrack for persistent multi-object tracking,
filtering for relevant dynamic classes: Person, Bicycle, Car, Motorcycle, Bus, Truck.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("visiontrack.detector")

# Filtered dynamic COCO classes
COCO_DYNAMIC_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class TrackedDetection:
    """Represents a single tracked object detection in a frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    def to_dict(self) -> Dict[str, Union[int, str, float, Tuple[float, float, float, float]]]:
        return asdict(self)


class ObjectDetectorTracker:
    """
    Wraps YOLOv8 tracking pipeline using ByteTrack for persistent ID association
    and COCO dynamic vehicle/pedestrian class filtering.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml",
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name: Ultralytics YOLO model weight file or name (e.g. 'yolov8n.pt').
            conf_threshold: Minimum detection confidence score.
            iou_threshold: NMS IoU threshold.
            tracker_config: Tracker configuration file ('bytetrack.yaml' or 'botsort.yaml').
            device: Computing device ('cpu', 'cuda', '0', etc.). None for automatic selection.
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-loads the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO

            logger.info("Loading YOLO model: %s ...", self.model_name)
            self._model = YOLO(self.model_name)
            logger.info("YOLO model %s loaded successfully.", self.model_name)
        return self._model

    def track(self, frame: np.ndarray) -> List[TrackedDetection]:
        """
        Processes a video frame, performing object detection and ByteTrack association.

        Args:
            frame: BGR numpy image array.

        Returns:
            List of TrackedDetection objects with active track IDs.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or invalid frame passed to detector.")
            return []

        # Run YOLO with ByteTrack association
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            classes=list(COCO_DYNAMIC_CLASSES.keys()),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )

        detections: List[TrackedDetection] = []
        if not results or len(results) == 0:
            return detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract numpy arrays safely
        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy().astype(int)

        # ByteTrack assigns tracking IDs when persist=True
        if boxes.id is not None:
            id_arr = boxes.id.int().cpu().numpy()
        else:
            # No tracks associated yet
            return detections

        for i in range(len(id_arr)):
            track_id = int(id_arr[i])
            class_id = int(cls_arr[i])
            confidence = float(conf_arr[i])
            bbox = (
                float(xyxy_arr[i][0]),
                float(xyxy_arr[i][1]),
                float(xyxy_arr[i][2]),
                float(xyxy_arr[i][3]),
            )
            class_name = COCO_DYNAMIC_CLASSES.get(class_id, f"class_{class_id}")

            detections.append(
                TrackedDetection(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        return detections

    def reset_tracker(self) -> None:
        """Resets tracker internal state between separate video streams."""
        if self._model is not None:
            try:
                # Reset ByteTrack state if available
                if hasattr(self._model, "predictor") and self._model.predictor is not None:
                    if hasattr(self._model.predictor, "trackers"):
                        for t in self._model.predictor.trackers:
                            t.reset()
            except Exception as e:
                logger.debug("Tracker reset notification: %s", e)
