from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import REPO_ROOT


def load_env() -> None:
    """Load repo-root .env then optional local overrides."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()


@dataclass(frozen=True)
class RoboflowConfig:
    api_key: str
    model_id: str
    confidence: float
    api_url: str
    class_filter: frozenset[str]
    object_classes: frozenset[str]
    gripper_classes: frozenset[str]


def _normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _class_set_from_env(key: str, default: str) -> frozenset[str]:
    raw = os.environ.get(key, default).strip()
    return frozenset(_normalize_class_name(item) for item in raw.split(",") if item.strip())


def get_roboflow_config() -> RoboflowConfig:
    load_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Copy .env.example to .env at the repo root."
        )

    raw_filter = os.environ.get("ROBOFLOW_CLASS_FILTER", "").strip()
    class_filter = frozenset(
        _normalize_class_name(item) for item in raw_filter.split(",") if item.strip()
    )

    return RoboflowConfig(
        api_key=api_key,
        model_id=os.environ.get("ROBOFLOW_MODEL_ID", "yolov8s-640").strip(),
        confidence=float(os.environ.get("ROBOFLOW_CONFIDENCE", "0.12")),
        api_url=os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com").strip(),
        class_filter=class_filter,
        object_classes=_class_set_from_env(
            "ROBOFLOW_OBJECT_CLASSES",
            "block,cube,black_cube",
        ),
        gripper_classes=_class_set_from_env(
            "ROBOFLOW_GRIPPER_CLASSES",
            "gripper_tip,yellow_tape,gripper",
        ),
    )
