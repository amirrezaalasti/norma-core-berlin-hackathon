from __future__ import annotations

import os

import cv2
import numpy as np

from .workspace import WorkspaceCalibration, _order_corners, detect_workspace_board


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _square_marker_candidates(
    image_rgb: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Find square-ish dark blobs (AprilTag / ArUco appearance) in image coordinates."""
    height, width = image_rgb.shape[:2]
    if max(height, width) < 900:
        image_rgb = cv2.resize(
            image_rgb,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )
        height, width = image_rgb.shape[:2]
        scale = 2.0
    else:
        scale = 1.0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    image_area = float(height * width)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float]] = []
    inv_scale = 1.0 / scale

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < max(image_area * 0.00015, 120.0) or area > image_area * 0.08:
            continue

        x, y, box_w, box_h = cv2.boundingRect(contour)
        aspect = max(box_w, box_h) / max(min(box_w, box_h), 1)
        if aspect > 1.8:
            continue

        fill_ratio = area / max(float(box_w * box_h), 1.0)
        if fill_ratio < 0.4:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.06 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue

        cx = float(moments["m10"] / moments["m00"]) * inv_scale
        cy = float(moments["m01"] / moments["m00"]) * inv_scale
        candidates.append((cx, cy, area * inv_scale * inv_scale))

    return candidates


def _validate_marker_quad(
    ordered: np.ndarray,
    board_corners: np.ndarray,
    width: int,
    height: int,
) -> bool:
    board_diag = float(np.linalg.norm(board_corners[0] - board_corners[2]))
    if board_diag <= 1.0:
        return False

    for index in range(4):
        dist = float(np.linalg.norm(ordered[index] - board_corners[index]))
        if dist > board_diag * 0.35:
            return False

    margin = min(width, height) * 0.02
    board_inset = all(
        margin < corner[0] < width - margin and margin < corner[1] < height - margin
        for corner in board_corners
    )
    if board_inset:
        for point in ordered:
            if point[0] <= margin or point[1] <= margin or point[0] >= width - margin or point[1] >= height - margin:
                return False

    quad_area = float(cv2.contourArea(ordered.reshape(-1, 1, 2).astype(np.float32)))
    image_area = float(width * height)
    if quad_area < image_area * 0.015 or quad_area > image_area * 0.75:
        return False

    width_px = float(np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])) / 2.0
    height_px = float(np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])) / 2.0
    aspect = max(width_px, height_px) / max(min(width_px, height_px), 1.0)
    return 1.1 <= aspect <= 3.5


def _pick_four_corners_board_guided(
    candidates: list[tuple[float, float, float]],
    board_corners: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray | None:
    if len(candidates) < 4:
        return None

    board_ordered = _order_corners(board_corners.astype(np.float32))
    board_width = float(np.linalg.norm(board_ordered[1] - board_ordered[0]))
    board_height = float(np.linalg.norm(board_ordered[3] - board_ordered[0]))
    search_radius = max(board_width, board_height) * 0.32
    image_area = float(width * height)
    tag_min_area = max(image_area * 0.00015, 80.0)
    tag_max_area = image_area * 0.012
    margin = min(width, height) * 0.025

    selected: list[tuple[float, float]] = []
    for corner in board_ordered:
        in_range = [
            candidate
            for candidate in candidates
            if float(np.hypot(candidate[0] - corner[0], candidate[1] - corner[1])) <= search_radius
            and margin <= candidate[0] <= width - margin
            and margin <= candidate[1] <= height - margin
        ]
        if not in_range:
            return None

        tag_like = [
            candidate
            for candidate in in_range
            if tag_min_area <= candidate[2] <= tag_max_area
        ]
        pool = tag_like if tag_like else in_range
        best = min(
            pool,
            key=lambda candidate: float(np.hypot(candidate[0] - corner[0], candidate[1] - corner[1])),
        )
        selected.append((best[0], best[1]))

    ordered = _order_corners(np.array(selected, dtype=np.float32))
    if not _validate_marker_quad(ordered, board_ordered, width, height):
        return None
    return ordered


def _opencv_marker_centers(image_rgb: np.ndarray) -> list[tuple[float, float]]:
    if not hasattr(cv2, "aruco"):
        return []

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    dictionary_names = (
        "DICT_APRILTAG_16H5",
        "DICT_APRILTAG_36H11",
        "DICT_APRILTAG_25H9",
        "DICT_APRILTAG_36H10",
    )
    centers: list[tuple[float, float]] = []

    for dictionary_name in dictionary_names:
        if not hasattr(cv2.aruco, dictionary_name):
            continue
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        parameters = cv2.aruco.DetectorParameters()
        parameters.minMarkerPerimeterRate = 0.001
        parameters.detectInvertedMarker = True

        for scale in (1.0, 1.5, 2.0):
            scaled = cv2.resize(clahe, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if hasattr(cv2.aruco, "ArucoDetector"):
                corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(scaled)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(scaled, dictionary, parameters=parameters)
            if ids is None:
                continue
            inv = 1.0 / scale
            for marker_corners in corners:
                points = marker_corners.reshape(4, 2)
                center = points.mean(axis=0)
                centers.append((float(center[0] * inv), float(center[1] * inv)))
            if len(centers) >= 4:
                return centers

    return centers


def _workspace_from_ordered_corners(
    ordered: np.ndarray,
    *,
    calibration_source: str,
    confidence: float,
    use_mm: bool,
    tag_ids: tuple[int, ...] | None = None,
    tag_family: str | None = None,
) -> WorkspaceCalibration:
    width_px = float(
        np.linalg.norm(ordered[1] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[3])
    ) / 2.0
    height_px = float(
        np.linalg.norm(ordered[3] - ordered[0]) + np.linalg.norm(ordered[2] - ordered[1])
    ) / 2.0
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
        confidence=confidence,
        origin_xy=(float(center[0]), float(center[1])),
        calibration_source=calibration_source,
        units="mm" if use_mm else "px",
        plane_width=board_width_mm if use_mm else width_px,
        plane_height=board_height_mm if use_mm else height_px,
        tag_inset_mm=tag_inset_mm if use_mm else None,
        tag_ids=tag_ids,
        tag_family=tag_family,
    )


def detect_workspace_four_markers(image_rgb: np.ndarray) -> WorkspaceCalibration | None:
    """Build homography from four corner fiducials (AprilTag centers or square markers)."""
    height, width = image_rgb.shape[:2]

    try:
        from .apriltag_workspace import detect_workspace_apriltags

        apriltag_workspace = detect_workspace_apriltags(image_rgb)
        if apriltag_workspace is not None:
            return apriltag_workspace
    except cv2.error:
        pass

    board = detect_workspace_board(image_rgb)
    if board is None:
        return None

    board_corners = np.array(board.corners_xy, dtype=np.float32)
    ordered: np.ndarray | None = None

    aruco_centers = _opencv_marker_centers(image_rgb)
    if len(aruco_centers) >= 4:
        ordered = _pick_four_corners_board_guided(
            [(x, y, 1.0) for x, y in aruco_centers],
            board_corners,
            width,
            height,
        )

    if ordered is None:
        square_candidates = _square_marker_candidates(image_rgb)
        ordered = _pick_four_corners_board_guided(square_candidates, board_corners, width, height)

    if ordered is None:
        return None

    return _workspace_from_ordered_corners(
        ordered,
        calibration_source="markers",
        confidence=0.88,
        use_mm=False,
    )
