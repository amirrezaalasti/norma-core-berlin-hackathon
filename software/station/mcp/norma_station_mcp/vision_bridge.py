from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from .paths import REPO_ROOT

VISION_ROOT = REPO_ROOT / "software" / "station" / "vision"
DEFAULT_CLASSES = ["cube", "block", "mug", "cup", "rectangular box"]
DEFAULT_MODEL = "yoloe-11s-seg.pt"


def _ensure_vision_importable() -> None:
    vision_path = str(VISION_ROOT)
    if vision_path not in sys.path:
        sys.path.insert(0, vision_path)


@lru_cache(maxsize=1)
def get_detector(model_name: str, classes_key: str, confidence: float):
    _ensure_vision_importable()
    try:
        from norma_vision.detector import ObjectDetector
    except ImportError as exc:
        raise ImportError(
            "Vision dependencies are missing. Install with: "
            "uv sync --project software/station/mcp --extra vision"
        ) from exc

    classes = [item.strip() for item in classes_key.split(",") if item.strip()]
    return ObjectDetector(
        model_name=model_name,
        classes=classes,
        confidence=confidence,
        device=os.environ.get("NORMA_VISION_DEVICE"),
    )


async def detect_from_station(
    classes: list[str],
    camera_index: int = 0,
    confidence: float = 0.25,
    model_name: str | None = None,
) -> dict:
    _ensure_vision_importable()
    try:
        from norma_vision.frames import fetch_camera_images
    except ImportError as exc:
        raise ImportError(
            "Vision dependencies are missing. Install with: "
            "uv sync --project software/station/mcp --extra vision"
        ) from exc

    from .session import get_session

    session = get_session()
    await session.ensure_connected()

    model = model_name or os.environ.get("NORMA_VISION_MODEL", DEFAULT_MODEL)
    classes_key = ",".join(classes)
    detector = get_detector(model, classes_key, confidence)

    rgb, meta = await fetch_camera_images(session.host, camera_index=camera_index)
    detections = detector.detect(rgb)

    return {
        **meta,
        "model": model,
        "classes": classes,
        "detection_count": len(detections),
        "detections": [item.to_dict() for item in detections],
    }
