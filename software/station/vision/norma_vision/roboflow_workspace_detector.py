from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import numpy as np

from .env_config import _normalize_class_name, get_roboflow_config
from .manual_workspace_store import load_manual_workspace
from .roboflow_detector import RoboflowDetector
from .types import Detection
from .workspace import (
    WorkspaceCalibration,
    enrich_detections_with_workspace,
    filter_detections_in_workspace,
    gripper_tip_position,
    pixel_to_board_normalized,
    pixel_to_board_offset,
    scale_workspace_to_image,
)


def _parse_ref_size(raw: str) -> tuple[float, float]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        return 299.0, 224.0
    return float(parts[0]), float(parts[1])


def resolve_manual_workspace(image_rgb: np.ndarray) -> WorkspaceCalibration | None:
    manual = load_manual_workspace()
    if manual is None:
        return None

    height, width = image_rgb.shape[:2]
    ref_size = _parse_ref_size(os.environ.get("NORMA_WORKSPACE_REF_SIZE", "299,224"))
    return scale_workspace_to_image(
        manual,
        width,
        height,
        reference_width=ref_size[0],
        reference_height=ref_size[1],
    )


def split_roboflow_detections(
    detections: list[Detection],
    object_classes: frozenset[str],
    gripper_classes: frozenset[str],
) -> tuple[list[Detection], list[Detection]]:
    objects: list[Detection] = []
    gripper: list[Detection] = []

    for detection in detections:
        normalized = _normalize_class_name(detection.class_name)
        if normalized in gripper_classes:
            gripper.append(detection)
        elif not object_classes or normalized in object_classes:
            objects.append(detection)

    return objects, gripper


def gripper_tip_from_detection(
    detection: Detection,
    workspace: WorkspaceCalibration,
) -> dict[str, Any]:
    px, py = detection.center_xy
    board_xy = pixel_to_board_normalized(px, py, workspace)
    offset = pixel_to_board_offset(px, py, workspace)
    return {
        "pixel_xy": [float(px), float(py)],
        "board_xy": list(board_xy) if board_xy is not None else None,
        "offset_xy": list(offset[0]) if offset is not None else [0.0, 0.0],
        "distance": offset[1] if offset is not None else 0.0,
        "class_name": detection.class_name,
        "confidence": float(detection.confidence),
        "source": "roboflow",
    }


def _detection_from_dict(item: dict[str, Any]) -> Detection:
    square_center = item.get("square_center_board_xy")
    square_local = item.get("square_local_xy")
    return Detection(
        class_name=str(item["class_name"]),
        confidence=float(item["confidence"]),
        bbox_xyxy=tuple(item["bbox_xyxy"]),
        center_xy=tuple(item["center_xy"]),
        size_wh=tuple(item["size_wh"]),
        angle_deg=float(item.get("angle_deg", 0.0)),
        board_xy=tuple(item["board_xy"]) if item.get("board_xy") else None,
        offset_xy=tuple(item["offset_xy"]) if item.get("offset_xy") else None,
        distance=item.get("distance"),
        square_id=item.get("square_id"),
        square_col=item.get("square_col"),
        square_row=item.get("square_row"),
        square_center_board_xy=tuple(square_center) if square_center else None,
        square_local_xy=tuple(square_local) if square_local else None,
    )


class RoboflowWorkspaceDetector:
    """Roboflow inference scoped to workspace objects + live gripper-tip tracking."""

    def __init__(self) -> None:
        self.config = get_roboflow_config()
        self.roboflow = RoboflowDetector(self.config)
        self.model_name = self.roboflow.model_name
        self.classes = sorted(self.config.object_classes)
        self.last_workspace: WorkspaceCalibration | None = None
        self.last_gripper_tip: dict[str, Any] | None = None
        self.last_gripper_detection: Detection | None = None
        self.last_calibration = None

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB uint8 image")

        workspace = resolve_manual_workspace(image_rgb)
        raw = self.roboflow.detect(image_rgb)
        objects, gripper_candidates = split_roboflow_detections(
            raw,
            self.config.object_classes,
            self.config.gripper_classes,
        )

        working_workspace = workspace
        gripper_tip: dict[str, Any] | None = None
        gripper_detection: Detection | None = None

        if gripper_candidates and workspace is not None:
            gripper_detection = max(gripper_candidates, key=lambda item: item.confidence)
            origin = gripper_detection.center_xy
            working_workspace = replace(
                workspace,
                origin_xy=origin,
                gripper_tip_set=True,
            )
            gripper_tip = gripper_tip_from_detection(gripper_detection, working_workspace)
        elif workspace is not None and workspace.gripper_tip_set:
            gripper_tip = gripper_tip_position(workspace)

        self.last_workspace = working_workspace
        self.last_gripper_tip = gripper_tip
        self.last_gripper_detection = gripper_detection

        if working_workspace is None:
            return objects[:5]

        dicts = enrich_detections_with_workspace(
            [item.to_dict() for item in objects],
            working_workspace,
        )
        dicts = filter_detections_in_workspace(dicts, working_workspace)
        dicts = [item for item in dicts if item.get("offset_xy") is not None]
        return [_detection_from_dict(item) for item in dicts[:5]]
