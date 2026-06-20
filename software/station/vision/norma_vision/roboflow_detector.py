from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .env_config import RoboflowConfig, get_roboflow_config
from .types import Detection


def _prediction_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        if not payload:
            return []
        payload = payload[0]
    if not isinstance(payload, dict):
        return []

    predictions = payload.get("predictions")
    if isinstance(predictions, list):
        return [item for item in predictions if isinstance(item, dict)]

    nested = payload.get("predictions", {})
    if isinstance(nested, dict):
        inner = nested.get("predictions")
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]

    return []


def _prediction_to_detection(item: dict[str, Any]) -> Detection | None:
    confidence = float(item.get("confidence", 0.0))
    class_name = str(item.get("class") or item.get("class_name") or "object")

    if "x" in item and "y" in item and "width" in item and "height" in item:
        cx = float(item["x"])
        cy = float(item["y"])
        width = float(item["width"])
        height = float(item["height"])
    elif "bbox" in item and isinstance(item["bbox"], dict):
        bbox = item["bbox"]
        x1 = float(bbox.get("x1", bbox.get("left", 0)))
        y1 = float(bbox.get("y1", bbox.get("top", 0)))
        x2 = float(bbox.get("x2", bbox.get("right", x1)))
        y2 = float(bbox.get("y2", bbox.get("bottom", y1)))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
    else:
        return None

    x1 = cx - width / 2.0
    y1 = cy - height / 2.0
    x2 = cx + width / 2.0
    y2 = cy + height / 2.0
    angle_deg = float(item.get("angle_deg", item.get("angle", 0.0)) or 0.0)

    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(x1, y1, x2, y2),
        center_xy=(cx, cy),
        size_wh=(width, height),
        angle_deg=angle_deg,
    )


class RoboflowDetector:
    """Hosted Roboflow inference via serverless API — fast, no local GPU weights."""

    def __init__(self, config: RoboflowConfig | None = None):
        try:
            from inference_sdk import InferenceConfiguration, InferenceHTTPClient
        except ImportError as exc:
            raise ImportError(
                "Roboflow inference requires inference-sdk. "
                "Install with: uv sync --project software/station/vision --extra roboflow"
            ) from exc

        self.config = config or get_roboflow_config()
        self.model_name = f"roboflow:{self.config.model_id}"
        self.client = InferenceHTTPClient(
            api_url=self.config.api_url,
            api_key=self.config.api_key,
        )
        self.client.configure(
            InferenceConfiguration(confidence_threshold=self.config.confidence)
        )

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB uint8 image")

        pil_image = Image.fromarray(image_rgb, mode="RGB")
        raw = self.client.infer(pil_image, model_id=self.config.model_id)
        detections: list[Detection] = []

        for item in _prediction_items(raw):
            detection = _prediction_to_detection(item)
            if detection is None:
                continue
            if self.config.class_filter and detection.class_name.lower() not in self.config.class_filter:
                continue
            detections.append(detection)

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections
