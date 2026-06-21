from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import os


@dataclass(frozen=True)
class WorkspaceCalibration:
    corners_xy: tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]
    center_xy: tuple[float, float]
    width_px: float
    height_px: float
    angle_deg: float
    confidence: float
    origin_xy: tuple[float, float] | None = None
    calibration_source: str = "board"
    units: str = "px"
    plane_width: float | None = None
    plane_height: float | None = None
    tag_inset_mm: float | None = None
    tag_ids: tuple[int, ...] | None = None
    tag_family: str | None = None
    gripper_tip_set: bool = False

    @property
    def offset_scale_x(self) -> float:
        if self.units == "mm" and self.plane_width is not None and self.tag_inset_mm is not None:
            return self.plane_width - 2.0 * self.tag_inset_mm
        return self.plane_width if self.plane_width is not None else self.width_px

    @property
    def offset_scale_y(self) -> float:
        if self.units == "mm" and self.plane_height is not None and self.tag_inset_mm is not None:
            return self.plane_height - 2.0 * self.tag_inset_mm
        return self.plane_height if self.plane_height is not None else self.height_px

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("origin_xy", "plane_width", "plane_height", "tag_inset_mm", "tag_ids", "tag_family"):
            if data.get(key) is None:
                data.pop(key, None)
        if self.calibration_source == "board":
            data.pop("calibration_source", None)
        if self.units == "px":
            data.pop("units", None)
        if not self.gripper_tip_set:
            data.pop("gripper_tip_set", None)
        return data


def _order_corners(points: np.ndarray) -> np.ndarray:
    ordered = points[np.argsort(points[:, 1])]
    top = ordered[:2][np.argsort(ordered[:2, 0])]
    bottom = ordered[2:][np.argsort(ordered[2:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def _build_board_mask(image_rgb: np.ndarray) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)
    lum = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    max_channel = pixels.max(axis=1)
    min_channel = pixels.min(axis=1)
    sat = np.divide(
        max_channel - min_channel,
        np.maximum(max_channel, 1.0),
    )
    board_flat = ((lum >= 110) & (lum <= 210) & (sat < 0.18)).astype(np.uint8)
    board_mask = board_flat.reshape(height, width) * 255

    # Erode fringe (gray halo around the board), then dilate back to the true board edge.
    radius = max(10, int(min(width, height) * 0.024))
    if radius % 2 == 0:
        radius += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius, radius))
    board_mask = cv2.erode(board_mask, kernel, iterations=1)
    board_mask = cv2.dilate(board_mask, kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    board_mask = cv2.morphologyEx(
        board_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    return board_mask


def _saturation(r: float, g: float, b: float) -> float:
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    if max_channel <= 0:
        return 0.0
    return (max_channel - min_channel) / max_channel


def _is_board_blue_dot_pixel(r: float, g: float, b: float) -> bool:
    """Faint blue calibration dots on the white board (not the gripper marker)."""
    sat = _saturation(r, g, b)
    return b > 75 and b > r + 8 and b >= g - 8 and sat > 0.12


def _is_blue_pixel(r: float, g: float, b: float) -> bool:
    sat = _saturation(r, g, b)
    return b > 70 and b > r * 1.05 and b > g * 0.95 and sat > 0.18


def detect_gripper_blue_dot(
    image_rgb: np.ndarray,
    board_mask: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Find the blue marker on the gripper tip (used as workspace origin)."""
    height, width = image_rgb.shape[:2]
    image_area = float(height * width)
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    blue_mask = np.zeros(height * width, dtype=np.uint8)
    for index, (r, g, b) in enumerate(pixels):
        if _is_blue_pixel(float(r), float(g), float(b)):
            blue_mask[index] = 255
    mask = blue_mask.reshape(height, width)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x = width / 2.0
    center_y = height / 2.0
    best_center: tuple[float, float] | None = None
    best_score = -1.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.00004 or area > image_area * 0.025:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        ix = int(round(cx))
        iy = int(round(cy))
        outside_board = (
            board_mask is not None
            and 0 <= ix < width
            and 0 <= iy < height
            and board_mask[iy, ix] == 0
        )
        dist = float(np.hypot(cx - center_x, cy - center_y))
        score = area + (2500.0 if outside_board else 0.0) - dist * 0.35
        if score > best_score:
            best_score = score
            best_center = (cx, cy)

    return best_center


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _pick_four_blue_dot_corners(
    candidates: list[tuple[float, float, float]],
) -> np.ndarray | None:
    if len(candidates) < 3:
        return None

    if len(candidates) == 3:
        points = np.array([(c[0], c[1]) for c in candidates], dtype=np.float32)
        ordered = _order_corners(points)
        d01 = float(np.linalg.norm(ordered[0] - ordered[1]))
        d12 = float(np.linalg.norm(ordered[1] - ordered[2]))
        d20 = float(np.linalg.norm(ordered[2] - ordered[0]))
        if d01 >= d12 and d01 >= d20:
            missing = ordered[0] + ordered[1] - ordered[2]
        elif d12 >= d01 and d12 >= d20:
            missing = ordered[1] + ordered[2] - ordered[0]
        else:
            missing = ordered[2] + ordered[0] - ordered[1]
        return _order_corners(np.vstack([ordered, missing]))

    if len(candidates) == 4:
        points = np.array([(c[0], c[1]) for c in candidates], dtype=np.float32)
        return _order_corners(points)

    from itertools import combinations

    best_ordered: np.ndarray | None = None
    best_area = -1.0
    for combo in combinations(candidates, 4):
        points = np.array([(c[0], c[1]) for c in combo], dtype=np.float32)
        ordered = _order_corners(points)
        quad_area = float(cv2.contourArea(ordered.reshape(-1, 1, 2).astype(np.float32)))
        if quad_area > best_area:
            best_area = quad_area
            best_ordered = ordered
    return best_ordered


def detect_board_blue_dots(image_rgb: np.ndarray) -> WorkspaceCalibration | None:
    """Homography from four blue calibration dots on the white board."""
    height, width = image_rgb.shape[:2]
    image_area = float(height * width)
    pixels = image_rgb.reshape(-1, 3).astype(np.float32)

    blue_mask = np.zeros(height * width, dtype=np.uint8)
    for index, (r, g, b) in enumerate(pixels):
        if _is_board_blue_dot_pixel(float(r), float(g), float(b)):
            blue_mask[index] = 255
    mask = blue_mask.reshape(height, width)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float]] = []
    max_area = min(image_area * 0.004, 600.0)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 10.0 or area > max_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        if cy < height * 0.18 or cy > height * 0.92:
            continue
        if cx < width * 0.08 or cx > width * 0.95:
            continue
        candidates.append((cx, cy, area))

    ordered = _pick_four_blue_dot_corners(candidates)
    if ordered is None:
        return None

    width_px = float(
        np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])
    ) / 2.0
    height_px = float(
        np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])
    ) / 2.0
    aspect = max(width_px, height_px) / max(min(width_px, height_px), 1.0)
    if aspect < 1.05 or aspect > 3.8:
        return None

    center = ordered.mean(axis=0)
    angle_deg = float(
        np.degrees(np.arctan2(ordered[1][1] - ordered[0][1], ordered[1][0] - ordered[0][0]))
    )

    board_width_mm = _env_float("NORMA_BOARD_WIDTH_MM", 280.0)
    board_height_mm = _env_float("NORMA_BOARD_HEIGHT_MM", 200.0)
    tag_inset_mm = _env_float("NORMA_TAG_INSET_MM", 25.0)

    return WorkspaceCalibration(
        corners_xy=(
            (float(ordered[0][0]), float(ordered[0][1])),
            (float(ordered[1][0]), float(ordered[1][1])),
            (float(ordered[2][0]), float(ordered[2][1])),
            (float(ordered[3][0]), float(ordered[3][1])),
        ),
        center_xy=(float(center[0]), float(center[1])),
        width_px=width_px,
        height_px=height_px,
        angle_deg=angle_deg,
        confidence=min(0.97, 0.78 + 0.04 * len(candidates)),
        origin_xy=(float(center[0]), float(center[1])),
        calibration_source="blue_dots",
        units="mm",
        plane_width=board_width_mm,
        plane_height=board_height_mm,
        tag_inset_mm=tag_inset_mm,
    )


def detect_workspace_board(image_rgb: np.ndarray) -> WorkspaceCalibration | None:
    height, width = image_rgb.shape[:2]
    image_area = float(height * width)
    board_mask = _build_board_mask(image_rgb)

    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_score = 0.0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.06 or area > image_area * 0.55:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        touches = sum([x <= 5, y <= 5, x + w >= width - 5, y + h >= height - 5])
        if touches >= 2:
            continue

        aspect = max(w, h) / max(min(w, h), 1)
        fill_ratio = area / max(float(w * h), 1.0)
        if aspect > 2.5 or fill_ratio < 0.45:
            continue

        score = area * (0.5 + fill_ratio)
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return None

    rect = cv2.minAreaRect(best_contour)
    corners = _order_corners(cv2.boxPoints(rect).astype(np.float32))
    center = corners.mean(axis=0)
    width_px = float(
        np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])
    ) / 2.0
    height_px = float(
        np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])
    ) / 2.0
    angle_deg = float(np.degrees(np.arctan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0])))

    x, y, w, h = cv2.boundingRect(best_contour)
    fill_ratio = float(cv2.contourArea(best_contour)) / max(float(w * h), 1.0)

    origin = None
    calibration_source = "board"
    if origin is None:
        origin = (float(center[0]), float(center[1]))

    return WorkspaceCalibration(
        corners_xy=(
            (float(corners[0][0]), float(corners[0][1])),
            (float(corners[1][0]), float(corners[1][1])),
            (float(corners[2][0]), float(corners[2][1])),
            (float(corners[3][0]), float(corners[3][1])),
        ),
        center_xy=(float(center[0]), float(center[1])),
        width_px=width_px,
        height_px=height_px,
        angle_deg=angle_deg,
        confidence=min(0.99, 0.55 + fill_ratio * 0.4),
        origin_xy=origin,
        calibration_source=calibration_source,
    )


def pixel_to_board_normalized(
    px: float,
    py: float,
    workspace: WorkspaceCalibration,
) -> tuple[float, float] | None:
    src = np.array(workspace.corners_xy, dtype=np.float32)

    if workspace.units == "mm" and workspace.tag_inset_mm is not None:
        inset = workspace.tag_inset_mm
        board_w = workspace.plane_width or 280.0
        board_h = workspace.plane_height or 200.0
        dst = np.array(
            [
                [inset, inset],
                [board_w - inset, inset],
                [board_w - inset, board_h - inset],
                [inset, board_h - inset],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(src, dst)
        point = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, homography)[0, 0]
        xmm, ymm = float(mapped[0]), float(mapped[1])
        usable_w = max(board_w - 2.0 * inset, 1.0)
        usable_h = max(board_h - 2.0 * inset, 1.0)
        u = (xmm - inset) / usable_w
        v = (ymm - inset) / usable_h
    else:
        dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        homography = cv2.getPerspectiveTransform(src, dst)
        point = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, homography)[0, 0]
        u, v = float(mapped[0]), float(mapped[1])

    if u < -0.05 or u > 1.05 or v < -0.05 or v > 1.05:
        return None
    return min(1.0, max(0.0, u)), min(1.0, max(0.0, v))


def pixel_to_board_plane(
    px: float,
    py: float,
    workspace: WorkspaceCalibration,
) -> tuple[float, float] | None:
    """Map image pixel to board-plane coordinates (mm when AprilTag-calibrated)."""
    board_xy = pixel_to_board_normalized(px, py, workspace)
    if board_xy is None:
        return None
    return board_xy[0] * workspace.offset_scale_x, board_xy[1] * workspace.offset_scale_y


def detect_workspace(image_rgb: np.ndarray) -> WorkspaceCalibration | None:
    from .manual_workspace_store import load_manual_workspace

    manual = load_manual_workspace()
    if manual is not None:
        return manual

    mode = os.environ.get("NORMA_WORKSPACE_MODE", "auto").strip().lower()

    if mode == "board":
        return detect_workspace_board(image_rgb)

    if mode == "blue_dots":
        return detect_board_blue_dots(image_rgb)

    if mode in ("apriltag", "markers"):
        from .marker_workspace import detect_workspace_four_markers

        return detect_workspace_four_markers(image_rgb)

    # auto: blue dots → AprilTags/square markers → white board outline
    blue_dots = detect_board_blue_dots(image_rgb)
    if blue_dots is not None:
        return blue_dots

    from .marker_workspace import detect_workspace_four_markers

    marker_workspace = detect_workspace_four_markers(image_rgb)
    if marker_workspace is not None:
        return marker_workspace

    return detect_workspace_board(image_rgb)


def scale_workspace_to_image(
    workspace: WorkspaceCalibration,
    image_width: int,
    image_height: int,
    *,
    reference_width: float | None = None,
    reference_height: float | None = None,
) -> WorkspaceCalibration:
    """Scale pixel coordinates when image resolution differs from calibration."""
    max_x = max(point[0] for point in workspace.corners_xy)
    max_y = max(point[1] for point in workspace.corners_xy)
    ref_w = reference_width or max(max_x * 1.08, 1.0)
    ref_h = reference_height or max(max_y * 1.08, 1.0)

    if image_width <= ref_w * 1.15 and image_height <= ref_h * 1.15:
        return workspace

    sx = image_width / ref_w
    sy = image_height / ref_h

    def scale_point(point: tuple[float, float]) -> tuple[float, float]:
        return point[0] * sx, point[1] * sy

    corners = tuple(scale_point(point) for point in workspace.corners_xy)
    center = scale_point(workspace.center_xy)
    origin = scale_point(workspace.origin_xy) if workspace.origin_xy else None

    return WorkspaceCalibration(
        corners_xy=corners,  # type: ignore[arg-type]
        center_xy=center,
        width_px=workspace.width_px * sx,
        height_px=workspace.height_px * sy,
        angle_deg=workspace.angle_deg,
        confidence=workspace.confidence,
        origin_xy=origin,
        calibration_source=workspace.calibration_source,
        units=workspace.units,
        plane_width=workspace.plane_width,
        plane_height=workspace.plane_height,
        tag_inset_mm=workspace.tag_inset_mm,
        tag_ids=workspace.tag_ids,
        tag_family=workspace.tag_family,
        gripper_tip_set=workspace.gripper_tip_set,
    )


def gripper_tip_position(
    workspace: WorkspaceCalibration,
) -> dict[str, Any] | None:
    """Board-plane position of the manual gripper tip (origin) from corner homography."""
    if not workspace.gripper_tip_set:
        return None
    origin = workspace.origin_xy or workspace.center_xy
    if origin is None:
        return None
    board_xy = pixel_to_board_normalized(origin[0], origin[1], workspace)
    offset = pixel_to_board_offset(origin[0], origin[1], workspace)
    return {
        "pixel_xy": [float(origin[0]), float(origin[1])],
        "board_xy": list(board_xy) if board_xy is not None else None,
        "offset_xy": list(offset[0]) if offset is not None else [0.0, 0.0],
        "distance": offset[1] if offset is not None else 0.0,
    }


def enrich_detection_with_workspace(
    detection: dict[str, Any],
    workspace: WorkspaceCalibration,
) -> dict[str, Any]:
    """Attach board_xy and gripper-relative offset using manual corner homography."""
    from .workspace_grid import square_info_from_board_xy

    center = detection.get("center_xy")
    if not center or len(center) < 2:
        return detection

    px, py = float(center[0]), float(center[1])
    board_xy = pixel_to_board_normalized(px, py, workspace)
    if workspace.calibration_source == "manual":
        can_offset = bool(workspace.corners_xy)
    else:
        can_offset = workspace.calibration_source != "manual" or workspace.gripper_tip_set
    offset = pixel_to_board_offset(px, py, workspace) if can_offset else None

    enriched = dict(detection)
    if board_xy is not None:
        enriched["board_xy"] = list(board_xy)
        square = square_info_from_board_xy(board_xy)
        enriched.update(square.to_dict())
    if offset is not None:
        enriched["offset_xy"] = list(offset[0])
        enriched["distance"] = float(offset[1])
    return enriched


def enrich_detections_with_workspace(
    detections: list[dict[str, Any]],
    workspace: WorkspaceCalibration,
) -> list[dict[str, Any]]:
    return [enrich_detection_with_workspace(item, workspace) for item in detections]


def filter_detections_in_workspace(
    detections: list[dict[str, Any]],
    workspace: WorkspaceCalibration,
    *,
    margin: float = 0.05,
) -> list[dict[str, Any]]:
    """Keep detections whose center maps inside the calibrated board (manual corners)."""
    filtered: list[dict[str, Any]] = []
    for detection in detections:
        center = detection.get("center_xy")
        if not center or len(center) < 2:
            continue
        board_xy = pixel_to_board_normalized(float(center[0]), float(center[1]), workspace)
        if board_xy is None:
            continue
        u, v = board_xy
        if u < -margin or u > 1.0 + margin or v < -margin or v > 1.0 + margin:
            continue
        filtered.append(detection)
    return filtered


def board_reference_pixel(workspace: WorkspaceCalibration) -> tuple[float, float]:
    if workspace.calibration_source == "manual":
        return workspace.center_xy
    return workspace.origin_xy or workspace.center_xy


def pixel_to_board_offset(
    px: float,
    py: float,
    workspace: WorkspaceCalibration,
) -> tuple[tuple[float, float], float] | None:
    obj_plane = pixel_to_board_plane(px, py, workspace)
    if obj_plane is None:
        return None

    origin = board_reference_pixel(workspace)
    origin_plane = pixel_to_board_plane(origin[0], origin[1], workspace)
    if origin_plane is None:
        return None

    dx = obj_plane[0] - origin_plane[0]
    dy = obj_plane[1] - origin_plane[1]
    return (dx, dy), float(np.hypot(dx, dy))
