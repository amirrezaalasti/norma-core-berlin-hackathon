from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .obb import obb_from_axis_aligned_box, obb_from_mask

DEFAULT_CLASSES = [
    "cube",
    "block",
    "mug",
    "cup",
    "rectangular box",
]

DEFAULT_MODEL = "yoloe-11s-seg.pt"
COCO_MODEL = "yolo11n.pt"


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    center_xy: tuple[float, float]
    size_wh: tuple[float, float]
    angle_deg: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["obb_xywha"] = [
            self.center_xy[0],
            self.center_xy[1],
            self.size_wh[0],
            self.size_wh[1],
            self.angle_deg,
        ]
        return data


class ObjectDetector:
    """Pretrained open-vocabulary detector — no custom training required.

    Default backend is YOLOE (segmentation + text prompts). Segmentation masks
    are converted to oriented bounding boxes via cv2.minAreaRect.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        classes: list[str] | None = None,
        confidence: float = 0.25,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.classes = list(classes or DEFAULT_CLASSES)
        self.confidence = confidence
        self.device = device
        self._model = None
        self._backend = self._resolve_backend(model_name)

    @staticmethod
    def _resolve_backend(model_name: str) -> str:
        lowered = model_name.lower()
        if "yoloe" in lowered or "world" in lowered:
            return "open_vocab"
        return "coco"

    def _load_model(self):
        if self._model is not None:
            return self._model

        from ultralytics import YOLO, YOLOE

        if self._backend == "open_vocab":
            if "yoloe" in self.model_name.lower():
                model = YOLOE(self.model_name)
            else:
                model = YOLO(self.model_name)
            model.set_classes(self.classes)
        else:
            model = YOLO(self.model_name)

        if self.device:
            model.to(self.device)

        self._model = model
        return model

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        """Run detection on an HxWx3 uint8 RGB image."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB uint8 image")

        model = self._load_model()
        results = model.predict(
            source=image_rgb,
            conf=self.confidence,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        names = result.names or {}
        detections: list[Detection] = []

        for index in range(len(boxes)):
            xyxy = boxes.xyxy[index].cpu().numpy()
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            confidence = float(boxes.conf[index].cpu().item())
            class_id = int(boxes.cls[index].cpu().item())
            class_name = str(names.get(class_id, class_id))

            obb = None
            if result.masks is not None and index < len(result.masks.data):
                mask = result.masks.data[index].cpu().numpy()
                if mask.shape[:2] != image_rgb.shape[:2]:
                    mask = _resize_mask(mask, image_rgb.shape[1], image_rgb.shape[0])
                obb = obb_from_mask(mask)

            if obb is None:
                obb = obb_from_axis_aligned_box(x1, y1, x2, y2)

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_xy=(obb["center_x"], obb["center_y"]),
                    size_wh=(obb["width"], obb["height"]),
                    angle_deg=obb["angle_deg"],
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
