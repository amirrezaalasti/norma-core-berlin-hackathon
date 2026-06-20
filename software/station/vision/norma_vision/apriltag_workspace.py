from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from .workspace import WorkspaceCalibration, _order_corners

# tagStandard41h12 is the AprilRobotics 3.x default; tag16h5 remains for older prints.
APRILTAG_FAMILIES = (
    "tagStandard41h12",
    "tagStandard52h13",
    "tag16h5",
    "tag36h11",
    "tag25h9",
    "tag36h10",
)

_OPENCV_FAMILIES: dict[str, str] = {
    "tag16h5": "DICT_APRILTAG_16H5",
    "tag25h9": "DICT_APRILTAG_25H9",
    "tag36h10": "DICT_APRILTAG_36H10",
    "tag36h11": "DICT_APRILTAG_36H11",
}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_tag_ids() -> list[int] | None:
    raw = os.environ.get("NORMA_APRILTAG_IDS", "").strip()
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _configured_families() -> tuple[str, ...]:
    raw = os.environ.get("NORMA_APRILTAG_FAMILY", "").strip()
    if raw:
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    return APRILTAG_FAMILIES


def _quad_decimate() -> float:
    return _env_float("NORMA_APRILTAG_QUAD_DECIMATE", 0.8)


def _opencv_detect(
    gray: np.ndarray,
    dictionary_name: str,
) -> list[tuple[int, tuple[float, float]]]:
    if not hasattr(cv2.aruco, dictionary_name):
        return []

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.minMarkerPerimeterRate = 0.001
    parameters.detectInvertedMarker = True
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    tagged: list[tuple[int, tuple[float, float]]] = []
    for scale in (1.0, 1.5, 2.0):
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(scaled)

        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(enhanced)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(enhanced, dictionary, parameters=parameters)

        if ids is None:
            continue

        inv_scale = 1.0 / scale
        for index, tag_id in enumerate(ids.flatten()):
            points = corners[index].reshape(4, 2)
            center = points.mean(axis=0)
            tagged.append((int(tag_id), (float(center[0] * inv_scale), float(center[1] * inv_scale))))

        if len(tagged) >= 4:
            break

    deduped: dict[int, tuple[float, float]] = {}
    for tag_id, center in tagged:
        deduped[tag_id] = center
    return list(deduped.items())


def _opencv_dict_name(family: str) -> str | None:
    return _OPENCV_FAMILIES.get(family)


def _official_apriltag_detect(gray: np.ndarray, family: str) -> list[tuple[int, tuple[float, float]]]:
    """AprilRobotics apriltag Python bindings (https://github.com/AprilRobotics/apriltag)."""
    try:
        from apriltag import apriltag as ApriltagDetector
    except ImportError:
        return []

    detector = ApriltagDetector(
        family,
        decimate=_quad_decimate(),
        refine_edges=True,
    )
    detections = detector.detect(gray.astype(np.uint8))
    deduped: dict[int, tuple[float, float]] = {}
    for det in detections:
        center = det["center"]
        deduped[int(det["id"])] = (float(center[0]), float(center[1]))
    return list(deduped.items())


def _pupil_detect(gray: np.ndarray, family: str) -> list[tuple[int, tuple[float, float]]]:
    try:
        from pupil_apriltags import Detector
    except ImportError:
        return []

    detector = Detector(
        families=family,
        nthreads=4,
        quad_decimate=_quad_decimate(),
        refine_edges=1,
        decode_sharpening=0.35,
    )
    tags = detector.detect(gray.astype(np.uint8))
    return [(int(tag.tag_id), (float(tag.center[0]), float(tag.center[1]))) for tag in tags]


def _prepare_gray_variants(gray: np.ndarray) -> list[tuple[float, np.ndarray]]:
    height, width = gray.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants: list[tuple[float, np.ndarray]] = []
    for scale in (1.0, 1.5, 2.0, 2.5):
        if max(height, width) * scale > 2400:
            continue
        if scale == 1.0:
            scaled = gray
        else:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append((scale, clahe.apply(scaled)))
    return variants


def _scale_tag_centers(
    tagged: list[tuple[int, tuple[float, float]]],
    scale: float,
) -> list[tuple[int, tuple[float, float]]]:
    if scale == 1.0:
        return tagged
    inv = 1.0 / scale
    return [(tag_id, (center[0] * inv, center[1] * inv)) for tag_id, center in tagged]


def _use_official_apriltag() -> bool:
    return os.environ.get("NORMA_APRILTAG_USE_OFFICIAL", "").strip().lower() in ("1", "true", "yes")


def _detect_apriltags(gray: np.ndarray) -> tuple[list[tuple[int, tuple[float, float]]], str | None]:
    best: list[tuple[int, tuple[float, float]]] = []
    best_family: str | None = None

    for family in _configured_families():
        for scale, enhanced in _prepare_gray_variants(gray):
            if _use_official_apriltag():
                official_tags = _scale_tag_centers(_official_apriltag_detect(enhanced, family), scale)
                if len(official_tags) > len(best):
                    best = official_tags
                    best_family = family
                if len(best) >= 4:
                    return best, best_family

            pupil_tags = _scale_tag_centers(_pupil_detect(enhanced, family), scale)
            if len(pupil_tags) > len(best):
                best = pupil_tags
                best_family = family
            if len(best) >= 4:
                return best, best_family

            opencv_name = _opencv_dict_name(family)
            if opencv_name is not None:
                opencv_tags = _scale_tag_centers(_opencv_detect(enhanced, opencv_name), scale)
                if len(opencv_tags) > len(best):
                    best = opencv_tags
                    best_family = family
                if len(best) >= 4:
                    return best, best_family

    if len(best) >= 3:
        return best, best_family
    return [], None


def _select_corner_tags(
    tagged: list[tuple[int, tuple[float, float]]],
) -> list[tuple[int, tuple[float, float]]]:
    configured_ids = _env_tag_ids()
    if configured_ids is not None:
        by_id = {tag_id: center for tag_id, center in tagged}
        selected = []
        for tag_id in configured_ids:
            if tag_id in by_id:
                selected.append((tag_id, by_id[tag_id]))
        if len(selected) >= 3:
            return selected
        return []

    if len(tagged) < 3:
        return []

    if len(tagged) == 3:
        return tagged

    if len(tagged) == 4:
        centers = np.array([center for _, center in tagged], dtype=np.float32)
        ordered = _order_corners(centers)
        ordered_list = [tuple(point) for point in ordered]
        selected: list[tuple[int, tuple[float, float]]] = []
        used: set[tuple[float, float]] = set()
        for target in ordered_list:
            best = min(
                tagged,
                key=lambda item: (item[1][0] - target[0]) ** 2 + (item[1][1] - target[1]) ** 2,
            )
            if best[1] not in used:
                used.add(best[1])
                selected.append(best)
        return selected[:4]

    points = np.array([center for _, center in tagged], dtype=np.float32)
    ordered = _order_corners(points)
    ordered_list = [(float(x), float(y)) for x, y in ordered]
    selected = []
    selected_centers: list[tuple[float, float]] = []
    for target in ordered_list:
        best = min(
            tagged,
            key=lambda item: (item[1][0] - target[0]) ** 2 + (item[1][1] - target[1]) ** 2,
        )
        if best[1] not in selected_centers:
            selected_centers.append(best[1])
            selected.append(best)
        if len(selected) == 4:
            break
    return selected[:4]


def detect_workspace_apriltags(image_rgb: np.ndarray) -> WorkspaceCalibration | None:
    """Build a mm-scaled workspace plane from four corner AprilTags (allows 3-tag reconstruction)."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    tagged, family = _detect_apriltags(gray)
    if not tagged or family is None:
        return None

    selected = _select_corner_tags(tagged)
    if len(selected) < 3 or len(selected) > 4:
        return None

    board_width_mm = _env_float("NORMA_BOARD_WIDTH_MM", 280.0)
    board_height_mm = _env_float("NORMA_BOARD_HEIGHT_MM", 200.0)
    tag_inset_mm = _env_float("NORMA_TAG_INSET_MM", 25.0)

    if configured_ids := _env_tag_ids():
        centers_by_id = {tag_id: center for tag_id, center in selected}
        missing_id = None
        for tag_id in configured_ids:
            if tag_id not in centers_by_id:
                missing_id = tag_id
                break
        if missing_id is not None:
            m = configured_ids.index(missing_id)
            opp = (m + 2) % 4
            adj1 = (m + 1) % 4
            adj2 = (m + 3) % 4
            opp_id = configured_ids[opp]
            adj1_id = configured_ids[adj1]
            adj2_id = configured_ids[adj2]
            p_opp = np.array(centers_by_id[opp_id], dtype=np.float32)
            p_adj1 = np.array(centers_by_id[adj1_id], dtype=np.float32)
            p_adj2 = np.array(centers_by_id[adj2_id], dtype=np.float32)
            p_m = p_adj1 + p_adj2 - p_opp
            centers_by_id[missing_id] = (float(p_m[0]), float(p_m[1]))
        ordered_centers = np.array(
            [centers_by_id[tag_id] for tag_id in configured_ids],
            dtype=np.float32,
        )
        tag_ids = tuple(configured_ids)
    else:
        if len(selected) == 3:
            pts = np.array([center for _, center in selected], dtype=np.float32)
            d01 = np.linalg.norm(pts[0] - pts[1])
            d12 = np.linalg.norm(pts[1] - pts[2])
            d20 = np.linalg.norm(pts[2] - pts[0])
            if d01 >= d12 and d01 >= d20:
                diag_a, diag_b, opp = pts[0], pts[1], pts[2]
            elif d12 >= d01 and d12 >= d20:
                diag_a, diag_b, opp = pts[1], pts[2], pts[0]
            else:
                diag_a, diag_b, opp = pts[2], pts[0], pts[1]
            missing = diag_a + diag_b - opp
            all_4 = np.vstack([pts, missing])
            ordered_centers = _order_corners(all_4)
        else:
            ordered_centers = _order_corners(
                np.array([center for _, center in selected], dtype=np.float32)
            )
        tag_ids = tuple(tag_id for tag_id, _ in selected)

    width_px = float(
        np.linalg.norm(ordered_centers[1] - ordered_centers[0])
        + np.linalg.norm(ordered_centers[2] - ordered_centers[3])
    ) / 2.0
    height_px = float(
        np.linalg.norm(ordered_centers[3] - ordered_centers[0])
        + np.linalg.norm(ordered_centers[2] - ordered_centers[1])
    ) / 2.0
    center = ordered_centers.mean(axis=0)
    angle_deg = float(
        np.degrees(
            np.arctan2(
                ordered_centers[1][1] - ordered_centers[0][1],
                ordered_centers[1][0] - ordered_centers[0][0],
            )
        )
    )

    origin_xy = (float(center[0]), float(center[1]))
    confidence = min(0.99, 0.75 + 0.04 * len(selected))

    return WorkspaceCalibration(
        corners_xy=(
            (float(ordered_centers[0][0]), float(ordered_centers[0][1])),
            (float(ordered_centers[1][0]), float(ordered_centers[1][1])),
            (float(ordered_centers[2][0]), float(ordered_centers[2][1])),
            (float(ordered_centers[3][0]), float(ordered_centers[3][1])),
        ),
        center_xy=(float(center[0]), float(center[1])),
        width_px=width_px,
        height_px=height_px,
        angle_deg=angle_deg,
        confidence=confidence,
        origin_xy=origin_xy,
        calibration_source="apriltag",
        units="mm",
        plane_width=board_width_mm,
        plane_height=board_height_mm,
        tag_inset_mm=tag_inset_mm,
        tag_ids=tag_ids,
        tag_family=family,
    )


def apriltag_config_summary() -> dict[str, Any]:
    return {
        "families_tried": list(_configured_families()),
        "board_width_mm": _env_float("NORMA_BOARD_WIDTH_MM", 280.0),
        "board_height_mm": _env_float("NORMA_BOARD_HEIGHT_MM", 200.0),
        "tag_inset_mm": _env_float("NORMA_TAG_INSET_MM", 25.0),
        "quad_decimate": _quad_decimate(),
        "tag_ids": _env_tag_ids(),
        "workspace_mode": os.environ.get("NORMA_WORKSPACE_MODE", "auto"),
    }
