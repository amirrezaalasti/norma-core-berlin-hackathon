from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

from .paths import REPO_ROOT

VISION_ROOT = REPO_ROOT / "software" / "station" / "vision"
DEFAULT_CLASSES = ["can", "blue bottle", "box", "cube", "block", "mug", "cup"]
DEFAULT_MODEL = "yolov8s-worldv2.pt"

logger = logging.getLogger("norma-station-mcp.vision")


def _ensure_vision_importable() -> None:
    vision_path = str(VISION_ROOT)
    if vision_path not in sys.path:
        sys.path.insert(0, vision_path)


def _load_env() -> None:
    _ensure_vision_importable()
    try:
        from norma_vision.env_config import load_env

        load_env()
    except ImportError:
        pass


def _vision_backend() -> str:
    _load_env()
    backend = os.environ.get("NORMA_VISION_BACKEND", "auto").strip().lower()
    if backend not in ("auto", "roboflow", "contrast"):
        return "auto"
    if backend == "auto" and not os.environ.get("ROBOFLOW_API_KEY", "").strip():
        return "contrast"
    return backend


def _roboflow_enabled() -> bool:
    _load_env()
    return bool(os.environ.get("ROBOFLOW_API_KEY", "").strip())


def _parse_ref_size(raw: str) -> tuple[float, float]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        return 299.0, 224.0
    return float(parts[0]), float(parts[1])


def _resolve_workspace(rgb):
    from norma_vision.manual_workspace_store import load_manual_workspace
    from norma_vision.workspace import scale_workspace_to_image

    workspace = load_manual_workspace()
    if workspace is None:
        return None

    height, width = rgb.shape[:2]
    ref_size = _parse_ref_size(os.environ.get("NORMA_WORKSPACE_REF_SIZE", "299,224"))
    return scale_workspace_to_image(
        workspace,
        width,
        height,
        reference_width=ref_size[0],
        reference_height=ref_size[1],
    )


def _detections_from_dicts(dicts: list[dict]) -> list:
    from norma_vision.types import Detection

    return [
        Detection(
            class_name=str(item["class_name"]),
            confidence=float(item["confidence"]),
            bbox_xyxy=tuple(item["bbox_xyxy"]),
            center_xy=tuple(item["center_xy"]),
            size_wh=tuple(item["size_wh"]),
            angle_deg=float(item.get("angle_deg", 0.0)),
            board_xy=tuple(item["board_xy"]) if item.get("board_xy") else None,
            offset_xy=tuple(item["offset_xy"]) if item.get("offset_xy") else None,
            distance=item.get("distance"),
        )
        for item in dicts
    ]


def _enrich_with_workspace(raw, workspace) -> list:
    from norma_vision.workspace import enrich_detections_with_workspace, filter_detections_in_workspace

    if workspace is None:
        return list(raw)

    dicts = enrich_detections_with_workspace([item.to_dict() for item in raw], workspace)
    if workspace.calibration_source == "manual":
        dicts = filter_detections_in_workspace(dicts, workspace)
    return _detections_from_dicts(dicts)


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


@lru_cache(maxsize=1)
def get_roboflow_detector():
    _ensure_vision_importable()
    from norma_vision.roboflow_detector import RoboflowDetector

    return RoboflowDetector()


@lru_cache(maxsize=1)
def get_contrast_detector(classes_key: str):
    _ensure_vision_importable()
    from norma_vision.contrast_detector import ContrastDetector

    classes = [item.strip() for item in classes_key.split(",") if item.strip()]
    return ContrastDetector(classes=classes or ["block"])


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

    rgb, meta = await fetch_camera_images(
        session.host,
        camera_index=camera_index,
        timeout_s=8.0,
        require_fresh=False,
    )
    detections = detector.detect(rgb)

    return {
        **meta,
        "model": model,
        "classes": classes,
        "detection_count": len(detections),
        "detections": [item.to_dict() for item in detections],
    }


def _yolo_fallback_enabled() -> bool:
    return os.environ.get("NORMA_VISION_YOLO_FALLBACK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _run_roboflow(rgb, workspace) -> tuple[list, str]:
    detector = get_roboflow_detector()
    raw = detector.detect(rgb)
    detections = _enrich_with_workspace(raw, workspace)
    return detections, detector.model_name


def _run_yolo_fallback(
    rgb,
    workspace,
    requested: list[str],
    classes_key: str,
) -> list:
    """YOLOE/YOLO-World when contrast finds nothing on low-res streams."""
    model = os.environ.get("NORMA_VISION_MODEL", DEFAULT_MODEL)
    confidence = float(os.environ.get("NORMA_VISION_YOLO_CONF", "0.12"))
    yolo = get_detector(model, classes_key, confidence)
    raw = yolo.detect(rgb)
    detections = _enrich_with_workspace(raw, workspace)
    return [item for item in detections if item.offset_xy is not None]


async def detect_workspace_objects(
    camera_index: int = 0,
    classes: list[str] | None = None,
) -> dict:
    """Roboflow / contrast vision with workspace board + gripper-origin offsets."""
    _load_env()
    _ensure_vision_importable()
    try:
        from norma_vision.frames import fetch_camera_images
        from norma_vision.workspace import gripper_tip_position
    except ImportError as exc:
        raise ImportError(
            "Vision dependencies are missing. Install with: "
            "uv sync --project software/station/mcp --extra vision"
        ) from exc

    from .session import get_session

    session = get_session()
    await session.ensure_connected()

    requested = classes or ["block"]
    classes_key = ",".join(requested)
    backend = _vision_backend()

    rgb, meta = await fetch_camera_images(
        session.host,
        camera_index=camera_index,
        timeout_s=8.0,
        require_fresh=False,
    )

    workspace = _resolve_workspace(rgb)
    gripper_tip = gripper_tip_position(workspace) if workspace is not None else None
    calibration = None
    detections: list = []
    model_name = backend

    if backend in ("auto", "roboflow") and _roboflow_enabled():
        try:
            detections, model_name = _run_roboflow(rgb, workspace)
            if detections:
                logger.info("Roboflow found %d object(s)", len(detections))
        except Exception as exc:
            logger.warning("Roboflow detection failed: %s", exc)

    if not detections and backend in ("auto", "contrast"):
        detector = get_contrast_detector(classes_key)
        detections = detector.detect(rgb)
        workspace = getattr(detector, "last_workspace", None) or workspace
        calibration = getattr(detector, "last_calibration", None)
        gripper_tip = getattr(detector, "last_gripper_tip", None) or gripper_tip
        model_name = detector.model_name

    if not detections and _yolo_fallback_enabled():
        try:
            detections = _run_yolo_fallback(rgb, workspace, requested, classes_key)
            if detections:
                model_name = os.environ.get("NORMA_VISION_MODEL", DEFAULT_MODEL)
                logger.info("YOLO fallback found %d object(s)", len(detections))
        except Exception as exc:
            logger.warning("YOLO fallback failed: %s", exc)

    return {
        **meta,
        "model": model_name,
        "classes": requested,
        "detection_count": len(detections),
        "detections": [item.to_dict() for item in detections],
        "workspace": workspace.to_dict() if workspace is not None else None,
        "gripper_tip": gripper_tip,
        "camera_calibration": calibration.to_dict() if calibration is not None else None,
    }
