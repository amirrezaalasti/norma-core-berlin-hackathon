from __future__ import annotations

import cv2
import numpy as np


def obb_from_mask(mask: np.ndarray, min_area: float = 100.0) -> dict[str, float] | None:
    """Fit a rotated rectangle to a binary segmentation mask."""
    binary = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None

    center, size, angle = cv2.minAreaRect(contour)
    width, height = size
    if width < height:
        width, height = height, width
        angle += 90.0

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "width": float(width),
        "height": float(height),
        "angle_deg": float(angle),
    }


def obb_from_axis_aligned_box(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    """Fallback OBB when only an axis-aligned box is available."""
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)
    return {
        "center_x": float((x1 + x2) / 2.0),
        "center_y": float((y1 + y2) / 2.0),
        "width": width,
        "height": height,
        "angle_deg": 0.0,
    }
