from __future__ import annotations

import numpy as np

from .camera_calibration import get_camera_calibration
from .contrast_fallback import detect_dark_objects
from .manual_workspace_store import load_manual_workspace
from .types import Detection
from .workspace import (
    WorkspaceCalibration,
    detect_workspace,
    enrich_detection_with_workspace,
    filter_detections_in_workspace,
    gripper_tip_position,
    pixel_to_board_normalized,
    pixel_to_board_offset,
)

DEFAULT_CLASSES = ["block"]

_CAMERA_MM_SOURCES = frozenset({"camera"})


def _apply_camera_mm_offsets(
    detection: Detection,
    workspace: WorkspaceCalibration,
    calibration,
) -> Detection:
    origin = workspace.origin_xy or workspace.center_xy
    if origin is None:
        return detection

    offset = calibration.pixel_offset_mm(
        detection.center_xy[0],
        detection.center_xy[1],
        origin[0],
        origin[1],
    )
    if offset is None:
        return detection

    offset_xy, distance = offset
    return Detection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox_xyxy=detection.bbox_xyxy,
        center_xy=detection.center_xy,
        size_wh=detection.size_wh,
        angle_deg=detection.angle_deg,
        board_xy=detection.board_xy,
        offset_xy=offset_xy,
        distance=distance,
    )


def _maybe_upgrade_workspace_units(
    workspace: WorkspaceCalibration,
    calibration,
) -> WorkspaceCalibration:
    if calibration is None or not calibration.has_extrinsics:
        return workspace
    if workspace.calibration_source in ("manual", "apriltag", "markers", "blue_dots"):
        return workspace
    if workspace.calibration_source == "camera":
        return workspace
    if workspace.origin_xy is None and not workspace.gripper_tip_set:
        return workspace
    return WorkspaceCalibration(
        corners_xy=workspace.corners_xy,
        center_xy=workspace.center_xy,
        width_px=workspace.width_px,
        height_px=workspace.height_px,
        angle_deg=workspace.angle_deg,
        confidence=workspace.confidence,
        origin_xy=workspace.origin_xy,
        calibration_source="camera",
        units="mm",
        plane_width=workspace.plane_width,
        plane_height=workspace.plane_height,
        tag_inset_mm=workspace.tag_inset_mm,
        tag_ids=workspace.tag_ids,
        tag_family=workspace.tag_family,
        gripper_tip_set=workspace.gripper_tip_set or workspace.origin_xy is not None,
    )


def _resolve_workspace(image_rgb: np.ndarray) -> WorkspaceCalibration | None:
    """Prefer saved manual 4-point calibration; fall back to auto-detect only when absent."""
    manual = load_manual_workspace()
    if manual is not None:
        return manual

    workspace = detect_workspace(image_rgb)
    if workspace is None:
        return None

    calibration = get_camera_calibration()
    return _maybe_upgrade_workspace_units(workspace, calibration)


def _enrich_detection(
    detection: Detection,
    workspace: WorkspaceCalibration,
    calibration,
) -> Detection:
    if workspace.calibration_source == "manual":
        payload = enrich_detection_with_workspace(detection.to_dict(), workspace)
        return Detection(
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox_xyxy=detection.bbox_xyxy,
            center_xy=detection.center_xy,
            size_wh=detection.size_wh,
            angle_deg=detection.angle_deg,
            board_xy=tuple(payload["board_xy"]) if payload.get("board_xy") else None,
            offset_xy=tuple(payload["offset_xy"]) if payload.get("offset_xy") else None,
            distance=payload.get("distance"),
        )

    board_xy = pixel_to_board_normalized(
        detection.center_xy[0],
        detection.center_xy[1],
        workspace,
    )
    can_offset = workspace.calibration_source != "manual" or workspace.gripper_tip_set
    use_camera_mm = (
        calibration is not None
        and calibration.has_extrinsics
        and workspace.calibration_source in _CAMERA_MM_SOURCES
        and can_offset
    )
    if use_camera_mm:
        enriched_detection = _apply_camera_mm_offsets(detection, workspace, calibration)
        return Detection(
            class_name=enriched_detection.class_name,
            confidence=enriched_detection.confidence,
            bbox_xyxy=enriched_detection.bbox_xyxy,
            center_xy=enriched_detection.center_xy,
            size_wh=enriched_detection.size_wh,
            angle_deg=enriched_detection.angle_deg,
            board_xy=board_xy,
            offset_xy=enriched_detection.offset_xy,
            distance=enriched_detection.distance,
        )

    offset = (
        pixel_to_board_offset(
            detection.center_xy[0],
            detection.center_xy[1],
            workspace,
        )
        if can_offset
        else None
    )
    offset_xy = offset[0] if offset is not None else None
    distance = offset[1] if offset is not None else None
    return Detection(
        class_name=detection.class_name,
        confidence=detection.confidence,
        bbox_xyxy=detection.bbox_xyxy,
        center_xy=detection.center_xy,
        size_wh=detection.size_wh,
        angle_deg=detection.angle_deg,
        board_xy=board_xy,
        offset_xy=offset_xy,
        distance=distance,
    )


class ContrastDetector:
    """Fast local vision for dark objects on bright surfaces — no ML weights."""

    def __init__(
        self,
        classes: list[str] | None = None,
        **_kwargs: object,
    ):
        self.classes = list(classes or DEFAULT_CLASSES)
        self.model_name = "local-contrast"
        self.last_workspace = None
        self.last_gripper_tip: dict | None = None
        self.last_calibration = get_camera_calibration()

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        class_name = self.classes[0] if self.classes else "black block"
        calibration = self.last_calibration or get_camera_calibration()

        detections = detect_dark_objects(image_rgb, class_name=class_name)
        workspace = _resolve_workspace(image_rgb)
        self.last_workspace = workspace
        self.last_gripper_tip = gripper_tip_position(workspace) if workspace is not None else None

        if workspace is None:
            return detections

        enriched = [_enrich_detection(detection, workspace, calibration) for detection in detections]
        if workspace.calibration_source == "manual":
            filtered_dicts = filter_detections_in_workspace(
                [item.to_dict() for item in enriched],
                workspace,
            )
            centers = {tuple(d["center_xy"]) for d in filtered_dicts}
            enriched = [item for item in enriched if tuple(item.center_xy) in centers]
        return enriched
