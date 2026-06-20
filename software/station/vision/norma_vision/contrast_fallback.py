from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


def _luminance(r: float, g: float, b: float) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def _saturation(r: float, g: float, b: float) -> float:
    max_channel = max(r, g, b)
    min_channel = min(r, g, b)
    if max_channel <= 0:
        return 0.0
    return (max_channel - min_channel) / max_channel


def _is_red_pixel(r: float, g: float, b: float) -> bool:
    return r > 110 and r > g * 1.35 and r > b * 1.25 and _saturation(r, g, b) > 0.3


def _is_blue_pixel(r: float, g: float, b: float) -> bool:
    return b > 90 and b > r * 1.15 and b > g * 1.05 and _saturation(r, g, b) > 0.25


def _is_table_pixel(r: float, g: float, b: float) -> bool:
    lum = _luminance(r, g, b)
    return lum > 190 and _saturation(r, g, b) < 0.12


def _sample_region_stats(image_rgb: np.ndarray) -> dict[str, float]:
    step = max(1, int(np.sqrt(image_rgb.shape[0] * image_rgb.shape[1]) // 8))
    sampled = image_rgb[::step, ::step]
    pixels = sampled.reshape(-1, 3).astype(np.float32)
    red_ratio = float(np.mean([_is_red_pixel(r, g, b) for r, g, b in pixels]))
    blue_ratio = float(np.mean([_is_blue_pixel(r, g, b) for r, g, b in pixels]))
    luminance = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    saturation = (pixels.max(axis=1) - pixels.min(axis=1)) / np.maximum(pixels.max(axis=1), 1)
    dark_ratio = float(np.mean((luminance < 80) & (saturation < 0.3)))
    mean_luminance = float(np.mean(luminance))
    return {
        "red_ratio": red_ratio,
        "blue_ratio": blue_ratio,
        "dark_ratio": dark_ratio,
        "mean_luminance": mean_luminance,
    }


def _has_red_bull_branding(stats: dict[str, float]) -> bool:
    if stats["red_ratio"] < 0.025 or stats["blue_ratio"] < 0.025:
        return False
    if stats["dark_ratio"] > 0.5 and stats["mean_luminance"] < 70:
        return False
    return stats["red_ratio"] + stats["blue_ratio"] >= 0.08


def _is_black_block(stats: dict[str, float]) -> bool:
    if stats["dark_ratio"] >= 0.42 and stats["mean_luminance"] < 85:
        return True
    return stats["mean_luminance"] < 65


def _contour_to_detection(
    contour,
    image_rgb: np.ndarray,
    class_name: str,
    confidence: float,
) -> Detection | None:
    area = float(cv2.contourArea(contour))
    height, width = image_rgb.shape[:2]
    image_area = float(height * width)
    if area < image_area * 0.002 or area > image_area * 0.35:
        return None

    rect = cv2.minAreaRect(contour)
    box_width, box_height = rect[1]
    if box_width < 4 or box_height < 4:
        return None

    aspect = max(box_width, box_height) / max(min(box_width, box_height), 1.0)
    if aspect > 6.0:
        return None

    center_x, center_y = rect[0]
    angle = float(rect[2])
    if box_width < box_height:
        box_width, box_height = box_height, box_width
        angle += 90.0

    box_points = cv2.boxPoints(rect)
    x_coords = box_points[:, 0]
    y_coords = box_points[:, 1]
    x1, y1, x2, y2 = (
        float(np.min(x_coords)),
        float(np.min(y_coords)),
        float(np.max(x_coords)),
        float(np.max(y_coords)),
    )

    fill_ratio = area / max((x2 - x1) * (y2 - y1), 1.0)
    if fill_ratio < 0.25:
        return None

    return Detection(
        class_name=class_name,
        confidence=min(0.99, confidence + fill_ratio * 0.05),
        bbox_xyxy=(x1, y1, x2, y2),
        center_xy=(float(center_x), float(center_y)),
        size_wh=(float(box_width), float(box_height)),
        angle_deg=angle,
    )


def _grow_can_from_brand_seed(
    seed_contour,
    image_rgb: np.ndarray,
    gray: np.ndarray,
    brand_mask: np.ndarray,
) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    moments = cv2.moments(seed_contour)
    if moments["m00"] <= 0:
        return None

    seed_cx = moments["m10"] / moments["m00"]
    seed_cy = moments["m01"] / moments["m00"]
    max_radius = max(width, height) * 0.14

    x, y, w, h = cv2.boundingRect(seed_contour)
    region_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(region_mask, [seed_contour], -1, 255, thickness=-1)

    expanded = np.zeros((height, width), dtype=np.uint8)
    stack = np.column_stack(np.where(region_mask > 0)).tolist()
    visited = np.zeros((height, width), dtype=np.uint8)

    for cy, cx in stack:
        visited[cy, cx] = 1

    while stack:
        cy, cx = stack.pop()
        dx = cx - seed_cx
        dy = cy - seed_cy
        if dx * dx + dy * dy > max_radius * max_radius:
            continue

        expanded[cy, cx] = 255
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height or visited[ny, nx]:
                continue

            r, g, b = (float(v) for v in image_rgb[ny, nx])
            if _is_table_pixel(r, g, b):
                continue

            lum = float(gray[ny, nx])
            is_brand = brand_mask[ny, nx] > 0
            is_can_body = 28 < lum < 185
            if not is_brand and not is_can_body:
                continue

            visited[ny, nx] = 1
            stack.append((ny, nx))

    contours, _ = cv2.findContours(expanded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    x1 = max(0, int(np.min(contour[:, 0, 0])))
    y1 = max(0, int(np.min(contour[:, 0, 1])))
    x2 = min(width, int(np.ceil(np.max(contour[:, 0, 0]))))
    y2 = min(height, int(np.ceil(np.max(contour[:, 0, 1]))))
    stats = _sample_region_stats(image_rgb[y1:y2, x1:x2])
    if not _has_red_bull_branding(stats):
        return None
    return contour


def detect_dark_objects(
    image_rgb: np.ndarray,
    class_name: str = "black block",
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.35,
) -> list[Detection]:
    """Find objects on a bright table using contrast + color classification."""
    height, width = image_rgb.shape[:2]
    image_area = float(height * width)
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    red_channel = image_rgb[:, :, 0].astype(np.float32)
    green_channel = image_rgb[:, :, 1].astype(np.float32)
    blue_channel = image_rgb[:, :, 2].astype(np.float32)
    brand_mask = np.zeros((height, width), dtype=np.uint8)
    brand_mask[
        (red_channel > 110)
        & (red_channel > green_channel * 1.35)
        & (red_channel > blue_channel * 1.25)
    ] = 255
    brand_mask[
        (blue_channel > 90)
        & (blue_channel > red_channel * 1.15)
        & (blue_channel > green_channel * 1.05)
    ] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    detections: list[Detection] = []

    brand_clean = cv2.morphologyEx(brand_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    brand_clean = cv2.morphologyEx(brand_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    brand_contours, _ = cv2.findContours(brand_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for seed in brand_contours:
        if float(cv2.contourArea(seed)) < min_area * 0.15:
            continue
        contour = _grow_can_from_brand_seed(seed, image_rgb, gray, brand_mask)
        if contour is None:
            continue
        x1 = max(0, int(np.min(contour[:, 0, 0])))
        y1 = max(0, int(np.min(contour[:, 0, 1])))
        x2 = min(width, int(np.ceil(np.max(contour[:, 0, 0]))))
        y2 = min(height, int(np.ceil(np.max(contour[:, 0, 1]))))
        stats = _sample_region_stats(image_rgb[y1:y2, x1:x2])
        detection = _contour_to_detection(
            contour,
            image_rgb,
            "red bull",
            min(0.97, 0.65 + (stats["red_ratio"] + stats["blue_ratio"]) * 1.5),
        )
        if detection is not None:
            detections.append(detection)

    dark_clean = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_clean = cv2.morphologyEx(dark_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark_contours, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in dark_contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x1 = max(0, int(np.min(contour[:, 0, 0])))
        y1 = max(0, int(np.min(contour[:, 0, 1])))
        x2 = min(width, int(np.ceil(np.max(contour[:, 0, 0]))))
        y2 = min(height, int(np.ceil(np.max(contour[:, 0, 1]))))
        stats = _sample_region_stats(image_rgb[y1:y2, x1:x2])
        if not _is_black_block(stats) or _has_red_bull_branding(stats):
            continue

        detection = _contour_to_detection(
            contour,
            image_rgb,
            class_name if class_name in {"black block", "object"} else "black block",
            min(0.97, 0.6 + stats["dark_ratio"] * 0.35),
        )
        if detection is None:
            continue
        if any(_bbox_iou(existing.bbox_xyxy, detection.bbox_xyxy) > 0.15 for existing in detections):
            continue
        detections.append(detection)

    detections.sort(
        key=lambda item: (
            ((item.center_xy[0] - width / 2) ** 2 + (item.center_xy[1] - height / 2) ** 2),
            -(item.size_wh[0] * item.size_wh[1]),
        )
    )
    return _dedupe_detections(detections)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / (area_a + area_b - intersection)


def _dedupe_detections(detections: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if any(
            existing.class_name == detection.class_name
            and _bbox_iou(existing.bbox_xyxy, detection.bbox_xyxy) > 0.3
            for existing in kept
        ):
            continue
        kept.append(detection)
    return kept
