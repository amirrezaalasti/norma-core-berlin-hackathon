from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .paths import REPO_ROOT


def _default_intrinsics_path() -> Path:
    return Path(os.environ.get("NORMA_INTRINSICS_PATH", str(REPO_ROOT / "images" / "intrinsics.json")))


def _default_extrinsics_path() -> Path:
    return Path(os.environ.get("NORMA_EXTRINSICS_PATH", str(REPO_ROOT / "images" / "extrinsics.json")))


def _variant_name() -> str:
    return os.environ.get("NORMA_INTRINSICS_VARIANT", "aligned").strip().lower()


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    resolution: tuple[int, int]
    T_cam2world: np.ndarray | None = None
    board_plane_z_mm: float = 0.0
    reprojection_error_px: float | None = None
    source_paths: tuple[str, str] | None = None

    @property
    def has_extrinsics(self) -> bool:
        return self.T_cam2world is not None

    def undistort_image(self, image_rgb: np.ndarray) -> np.ndarray:
        return cv2.undistort(
            image_rgb,
            self.camera_matrix,
            self.distortion_coefficients,
        )

    def undistort_point(self, px: float, py: float) -> tuple[float, float]:
        pts = cv2.undistortPoints(
            np.array([[[px, py]]], dtype=np.float64),
            self.camera_matrix,
            self.distortion_coefficients,
            P=self.camera_matrix,
        )
        return float(pts[0, 0, 0]), float(pts[0, 0, 1])

    def pixel_to_plane_mm(self, px: float, py: float) -> tuple[float, float] | None:
        if self.T_cam2world is None:
            return None

        pts = cv2.undistortPoints(
            np.array([[[px, py]]], dtype=np.float64),
            self.camera_matrix,
            self.distortion_coefficients,
        )
        ray_cam = np.array([pts[0, 0, 0], pts[0, 0, 1], 1.0], dtype=np.float64)
        norm = np.linalg.norm(ray_cam)
        if norm <= 1e-9:
            return None
        ray_cam /= norm

        rotation = self.T_cam2world[:3, :3]
        origin = self.T_cam2world[:3, 3]
        direction = rotation @ ray_cam
        if abs(direction[2]) < 1e-9:
            return None

        scale = (self.board_plane_z_mm - origin[2]) / direction[2]
        if scale < 0:
            return None

        point = origin + scale * direction
        return float(point[0]), float(point[1])

    def pixel_offset_mm(
        self,
        px: float,
        py: float,
        origin_px: float,
        origin_py: float,
    ) -> tuple[tuple[float, float], float] | None:
        obj = self.pixel_to_plane_mm(px, py)
        origin = self.pixel_to_plane_mm(origin_px, origin_py)
        if obj is None or origin is None:
            return None
        dx = obj[0] - origin[0]
        dy = obj[1] - origin[1]
        return (dx, dy), float(np.hypot(dx, dy))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "resolution": list(self.resolution),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.reshape(-1).tolist(),
            "has_extrinsics": self.has_extrinsics,
            "board_plane_z_mm": self.board_plane_z_mm,
        }
        if self.reprojection_error_px is not None:
            payload["reprojection_error_px"] = self.reprojection_error_px
        if self.T_cam2world is not None:
            payload["T_cam2world"] = self.T_cam2world.tolist()
        if self.source_paths is not None:
            payload["source_paths"] = list(self.source_paths)
        return payload


def _load_intrinsics_variant(data: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant in data and isinstance(data[variant], dict):
        return data[variant]
    if "camera_matrix" in data:
        return data
    for key in ("aligned", "unaligned"):
        if key in data and isinstance(data[key], dict):
            return data[key]
    raise ValueError("intrinsics JSON is missing camera_matrix data")


def load_camera_calibration(
    intrinsics_path: Path | None = None,
    extrinsics_path: Path | None = None,
    variant: str | None = None,
) -> CameraCalibration | None:
    intrinsics_file = intrinsics_path or _default_intrinsics_path()
    extrinsics_file = extrinsics_path or _default_extrinsics_path()
    if not intrinsics_file.is_file():
        return None

    intrinsics_data = json.loads(intrinsics_file.read_text())
    intrinsics_variant = _load_intrinsics_variant(intrinsics_data, variant or _variant_name())
    camera_matrix = np.array(intrinsics_variant["camera_matrix"], dtype=np.float64)
    distortion = np.array(intrinsics_variant["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)

    resolution_raw = intrinsics_data.get("resolution")
    if resolution_raw and len(resolution_raw) == 2:
        resolution = (int(resolution_raw[0]), int(resolution_raw[1]))
    else:
        resolution = (int(camera_matrix[0, 2] * 2), int(camera_matrix[1, 2] * 2))

    T_cam2world = None
    board_plane_z_mm = float(os.environ.get("NORMA_BOARD_PLANE_Z_MM", "0"))
    reproj_error = None

    if extrinsics_file.is_file():
        extrinsics_data = json.loads(extrinsics_file.read_text())
        T_raw = extrinsics_data.get("T_cam2world")
        if T_raw is not None:
            T_cam2world = np.array(T_raw, dtype=np.float64)
        reproj_error = extrinsics_data.get("reprojection_error_px")

    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        resolution=resolution,
        T_cam2world=T_cam2world,
        board_plane_z_mm=board_plane_z_mm,
        reprojection_error_px=float(reproj_error) if reproj_error is not None else None,
        source_paths=(str(intrinsics_file), str(extrinsics_file) if extrinsics_file.is_file() else ""),
    )


_calibration_lock = threading.Lock()
_calibration_cache: CameraCalibration | None | bool = False


def get_camera_calibration() -> CameraCalibration | None:
    global _calibration_cache

    with _calibration_lock:
        if _calibration_cache is not False:
            return _calibration_cache or None

        if os.environ.get("NORMA_CAMERA_CALIBRATION", "1").strip().lower() in ("0", "false", "no"):
            _calibration_cache = None
            return None

        _calibration_cache = load_camera_calibration()
        return _calibration_cache


def calibration_payload_for_api() -> dict[str, Any] | None:
    calibration = get_camera_calibration()
    if calibration is None:
        return None
    return calibration.to_dict()
