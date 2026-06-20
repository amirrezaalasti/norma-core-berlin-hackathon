from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT
from .workspace import WorkspaceCalibration

_manual_lock = threading.Lock()
_manual_cache: WorkspaceCalibration | None = None

DEFAULT_PATH = REPO_ROOT / ".norma" / "manual_workspace.json"

DEFAULT_MANUAL_WORKSPACE: dict[str, Any] = {
    "corners_xy": [
        [21.415977961432507, 181.1900826446281],
        [60.129476584022036, 36.22038567493114],
        [242.16528925619832, 34.1611570247934],
        [270.58264462809916, 170.0702479338843],
    ],
    "center_xy": [148.573347107438, 105.41046831955924],
    "width_px": 144.4490172070461,
    "height_px": 215.7310656528827,
    "angle_deg": -75.04832264607566,
    "confidence": 1,
    "origin_xy": [180.38842975206612, 137.53443526170798],
    "calibration_source": "manual",
    "units": "mm",
    "plane_width": 280,
    "plane_height": 200,
    "tag_inset_mm": 25,
    "gripper_tip_set": True,
}


def manual_workspace_path() -> Path:
    return Path(os.environ.get("NORMA_MANUAL_WORKSPACE_PATH", str(DEFAULT_PATH)))


def _workspace_from_dict(data: dict[str, Any]) -> WorkspaceCalibration | None:
    corners = data.get("corners_xy")
    if not corners or len(corners) != 4:
        return None

    center = data.get("center_xy")
    origin = data.get("origin_xy")
    if not center or not origin:
        return None

    return WorkspaceCalibration(
        corners_xy=tuple(tuple(point) for point in corners),
        center_xy=tuple(center),
        width_px=float(data.get("width_px", 0.0)),
        height_px=float(data.get("height_px", 0.0)),
        angle_deg=float(data.get("angle_deg", 0.0)),
        confidence=float(data.get("confidence", 1.0)),
        origin_xy=tuple(origin),
        calibration_source=str(data.get("calibration_source", "manual")),
        units=str(data.get("units", "mm")),
        plane_width=data.get("plane_width"),
        plane_height=data.get("plane_height"),
        tag_inset_mm=data.get("tag_inset_mm"),
        gripper_tip_set=bool(data.get("gripper_tip_set")),
    )


def load_manual_workspace() -> WorkspaceCalibration | None:
    global _manual_cache

    with _manual_lock:
        if _manual_cache is not None:
            return _manual_cache

        path = manual_workspace_path()
        if path.is_file():
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                data = DEFAULT_MANUAL_WORKSPACE
        else:
            data = DEFAULT_MANUAL_WORKSPACE

        workspace = _workspace_from_dict(data)
        if workspace is None or not data.get("gripper_tip_set"):
            return None

        _manual_cache = workspace
        return workspace


def save_manual_workspace(data: dict[str, Any]) -> WorkspaceCalibration:
    global _manual_cache

    workspace = _workspace_from_dict(data)
    if workspace is None:
        raise ValueError("Invalid manual workspace payload")

    path = manual_workspace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

    with _manual_lock:
        _manual_cache = workspace if data.get("gripper_tip_set") else None

    return workspace


def clear_manual_workspace() -> None:
    global _manual_cache

    path = manual_workspace_path()
    if path.is_file():
        path.unlink()

    with _manual_lock:
        _manual_cache = None


def manual_workspace_ready(data: dict[str, Any]) -> bool:
    return bool(data.get("gripper_tip_set") and data.get("corners_xy"))
