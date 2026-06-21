from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

PICK_CALIBRATION_PATH = Path(
    os.environ.get(
        "NORMA_PICK_CALIBRATION_PATH",
        str(REPO_ROOT / ".norma" / "pick_calibration.json"),
    )
)


def load_pick_calibration() -> dict[str, Any] | None:
    if not PICK_CALIBRATION_PATH.is_file():
        return None
    return json.loads(PICK_CALIBRATION_PATH.read_text())


def save_pick_calibration(payload: dict[str, Any]) -> dict[str, Any]:
    PICK_CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    PICK_CALIBRATION_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def board_xy_to_offset_mm(
    board_xy: tuple[float, float],
    workspace: dict[str, Any],
) -> tuple[float, float]:
    """Convert normalized board coordinates to gripper-relative mm offset."""
    inset = float(workspace.get("tag_inset_mm") or 25.0)
    plane_width = float(workspace.get("plane_width") or 280.0)
    plane_height = float(workspace.get("plane_height") or 200.0)
    usable_w = max(plane_width - 2.0 * inset, 1.0)
    usable_h = max(plane_height - 2.0 * inset, 1.0)

    origin = workspace.get("origin_xy") or workspace.get("center_xy")
    if not origin:
        raise ValueError("workspace is missing origin_xy/center_xy")

    corners = workspace.get("corners_xy")
    if not corners or len(corners) != 4:
        raise ValueError("workspace is missing corners_xy")

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise ImportError("OpenCV is required for board_xy_to_offset_mm") from exc

    src = np.array(corners, dtype=np.float32)
    dst = np.array(
        [
            [inset, inset],
            [plane_width - inset, inset],
            [plane_width - inset, plane_height - inset],
            [inset, plane_height - inset],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, dst)

    def to_plane(px: float, py: float) -> tuple[float, float]:
        mapped = cv2.perspectiveTransform(
            np.array([[[px, py]]], dtype=np.float32),
            homography,
        )[0, 0]
        return float(mapped[0]), float(mapped[1])

    origin_plane = to_plane(float(origin[0]), float(origin[1]))
    obj_plane = (
        inset + board_xy[0] * usable_w,
        inset + board_xy[1] * usable_h,
    )
    return obj_plane[0] - origin_plane[0], obj_plane[1] - origin_plane[1]


# Per-joint board-axis coupling for single-point calibration (ElRobot).
# X = board +x offset (reach across board), Y = board +y offset (toward/away on board).
ELROBOT_JOINT_AXIS: dict[int, str] = {
    1: "x",
    2: "x",
    3: "x",
    4: "x",
    5: "y",
    6: "y",
    7: "y",
}

SO101_JOINT_AXIS: dict[int, str] = {
    1: "x",
    2: "x",
    3: "x",
    4: "x",
    5: "y",
}


def _joint_axis_map(arm_type: str) -> dict[int, str]:
    if arm_type == "elrobot":
        return ELROBOT_JOINT_AXIS
    return SO101_JOINT_AXIS


def _lstsq_scales(
    offsets_mm: list[tuple[float, float]],
    deltas: list[float],
) -> dict[str, float]:
    """Fit joint_delta ~= scale_dx * offset_x + scale_dy * offset_y."""
    if len(offsets_mm) < 2:
        raise ValueError("At least two calibration samples are required")

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("NumPy is required for multi-point pick calibration") from exc

    matrix = np.array([[dx, dy] for dx, dy in offsets_mm], dtype=np.float64)
    target = np.array(deltas, dtype=np.float64)
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return {"dx": float(solution[0]), "dy": float(solution[1])}


def derive_pick_scales_multi(
    home_joints: dict[int, float],
    samples: list[dict[str, Any]],
) -> dict[int, dict[str, float]]:
    """Fit per-joint 2D scales from multiple (offset_mm, pick_joints) samples."""
    if len(samples) < 2:
        raise ValueError("At least two calibration samples are required")

    scales: dict[int, dict[str, float]] = {}
    joint_ids = sorted(
        {
            joint_id
            for sample in samples
            for joint_id in sample["pick_joints"].keys()
            if joint_id in home_joints
        }
    )

    for joint_id in joint_ids:
        home_pos = float(home_joints[joint_id])
        offsets_mm: list[tuple[float, float]] = []
        deltas: list[float] = []

        for sample in samples:
            pick_joints = sample["pick_joints"]
            if joint_id not in pick_joints:
                continue
            pick_pos = float(pick_joints[joint_id])
            if pick_pos < -0.05 or pick_pos > 1.05:
                continue

            offset = sample["offset_mm"]
            offsets_mm.append((float(offset[0]), float(offset[1])))
            deltas.append(pick_pos - home_pos)

        if len(offsets_mm) < 2:
            continue
        scales[joint_id] = _lstsq_scales(offsets_mm, deltas)

    if not scales:
        raise ValueError("No valid joint samples found for multi-point calibration")
    return scales


def derive_pick_scales(
    home_joints: dict[int, float],
    pick_joints: dict[int, float],
    offset_mm: tuple[float, float],
    arm_type: str = "elrobot",
) -> dict[int, dict[str, float]]:
    dx, dy = offset_mm
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        raise ValueError("reference offset_mm must not be zero")

    axis_map = _joint_axis_map(arm_type)
    scales: dict[int, dict[str, float]] = {}
    for joint_id, home_pos in home_joints.items():
        delta = float(pick_joints.get(joint_id, home_pos)) - float(home_pos)
        axis = axis_map.get(joint_id, "x")
        if axis == "y":
            scales[joint_id] = {
                "dx": 0.0,
                "dy": delta / dy if abs(dy) > 1e-6 else 0.0,
            }
        else:
            scales[joint_id] = {
                "dx": delta / dx if abs(dx) > 1e-6 else 0.0,
                "dy": 0.0,
            }
    return scales


def _serialize_sample(
    *,
    board_xy: tuple[float, float],
    offset_mm: tuple[float, float],
    pick_joints: dict[int, float],
    label: str | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "board_xy": [round(board_xy[0], 4), round(board_xy[1], 4)],
        "offset_mm": [round(offset_mm[0], 2), round(offset_mm[1], 2)],
        "pick_joint_positions": {
            str(joint_id): round(value, 4) for joint_id, value in pick_joints.items()
        },
    }


def pick_calibration_from_poses(
    *,
    bus_serial: str,
    arm_type: str,
    home_joints: dict[int, float],
    pick_joints: dict[int, float],
    gripper_home: float | None,
    gripper_pick: float | None,
    offset_mm: tuple[float, float],
    board_xy: tuple[float, float] | None = None,
    workspace: dict[str, Any] | None = None,
    units: str = "mm",
    calibration_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    distance = math.hypot(offset_mm[0], offset_mm[1])
    samples = calibration_samples or []
    if not samples and board_xy is not None:
        samples = [
            _serialize_sample(
                board_xy=board_xy,
                offset_mm=offset_mm,
                pick_joints=pick_joints,
                label="reference",
            )
        ]

    parsed_samples = [
        {
            "offset_mm": tuple(float(v) for v in sample["offset_mm"]),
            "pick_joints": {
                int(joint_id): float(value)
                for joint_id, value in sample["pick_joint_positions"].items()
            },
        }
        for sample in samples
        if sample.get("label") != "home_tip"
    ]
    if len(parsed_samples) >= 2:
        scales = derive_pick_scales_multi(home_joints, parsed_samples)
        planning_mode = "multi_point_2d"
    else:
        scales = derive_pick_scales(home_joints, pick_joints, offset_mm, arm_type=arm_type)
        planning_mode = "single_point_axis"

    return {
        "bus_serial": bus_serial,
        "arm_type": arm_type,
        "units": units,
        "planning_mode": planning_mode,
        "home_joint_positions": {str(k): round(v, 4) for k, v in home_joints.items()},
        "reference_pick_joint_positions": {str(k): round(v, 4) for k, v in pick_joints.items()},
        "reference_offset_mm": [round(offset_mm[0], 2), round(offset_mm[1], 2)],
        "reference_distance_mm": round(distance, 2),
        "reference_board_xy": list(board_xy) if board_xy is not None else None,
        "calibration_samples": samples,
        "pick_scales": {
            str(joint_id): {
                "dx": round(values["dx"], 6),
                "dy": round(values["dy"], 6),
            }
            for joint_id, values in scales.items()
        },
        "gripper_home": gripper_home,
        "gripper_pick": gripper_pick,
        "workspace": workspace,
        "note": (
            "Empirical pick calibration: joint_delta ~= pick_scales.dx * offset_x_mm "
            "+ pick_scales.dy * offset_y_mm from home pose."
        ),
    }


def rebuild_pick_calibration_from_samples(
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Recompute pick_scales from stored calibration_samples."""
    home_joints = {
        int(joint_id): float(value)
        for joint_id, value in calibration["home_joint_positions"].items()
    }
    samples = calibration.get("calibration_samples") or []
    parsed_samples = [
        {
            "offset_mm": tuple(float(v) for v in sample["offset_mm"]),
            "pick_joints": {
                int(joint_id): float(value)
                for joint_id, value in sample["pick_joint_positions"].items()
            },
        }
        for sample in samples
        if sample.get("label") != "home_tip"
    ]
    if len(parsed_samples) < 2:
        raise ValueError("At least two non-home calibration_samples are required")

    scales = derive_pick_scales_multi(home_joints, parsed_samples)
    updated = {
        **calibration,
        "planning_mode": "multi_point_2d",
        "pick_scales": {
            str(joint_id): {
                "dx": round(values["dx"], 6),
                "dy": round(values["dy"], 6),
            }
            for joint_id, values in scales.items()
        },
    }
    save_pick_calibration(updated)
    return updated


def _pick_samples_from_calibration(
    calibration: dict[str, Any],
) -> list[tuple[tuple[float, float], dict[int, float], str]]:
    parsed: list[tuple[tuple[float, float], dict[int, float], str]] = []
    for sample in calibration.get("calibration_samples") or []:
        if sample.get("label") == "home_tip":
            continue
        offset = sample.get("offset_mm")
        picks = sample.get("pick_joint_positions")
        if not offset or not picks:
            continue
        pick_joints = {int(joint_id): float(value) for joint_id, value in picks.items()}
        parsed.append(
            (
                (float(offset[0]), float(offset[1])),
                pick_joints,
                str(sample.get("label") or ""),
            )
        )
    return parsed


def nearest_sample_from_calibration(
    offset_xy: tuple[float, float],
    calibration: dict[str, Any],
) -> dict[str, Any] | None:
    """Find the nearest calibration sample to the target offset (excluding home_tip)."""
    samples = calibration.get("calibration_samples") or []
    valid_samples = [s for s in samples if s.get("label") != "home_tip" and s.get("offset_mm")]
    if not valid_samples:
        return None

    tx, ty = float(offset_xy[0]), float(offset_xy[1])
    nearest_sample = None
    nearest_dist = float("inf")
    for sample in valid_samples:
        ox, oy = float(sample["offset_mm"][0]), float(sample["offset_mm"][1])
        dist = math.hypot(tx - ox, ty - oy)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_sample = sample
    return nearest_sample


def joint_targets_from_calibration_samples(
    home_joints: dict[int, float],
    offset_xy: tuple[float, float],
    calibration: dict[str, Any],
    *,
    min_samples: int = 2,
    snap_distance_mm: float = 12.0,
    power: float = 2.0,
) -> dict[int, float] | None:
    """Map board-plane offset to joints via nearest sample or IDW over recordings."""
    planning_mode = calibration.get("planning_mode")
    parsed = _pick_samples_from_calibration(calibration)

    if planning_mode == "static_hardcoded":
        if not parsed:
            return None
        tx, ty = float(offset_xy[0]), float(offset_xy[1])
        nearest_joints: dict[int, float] | None = None
        nearest_dist = float("inf")
        for (ox, oy), pick_joints, _label in parsed:
            dist = math.hypot(tx - ox, ty - oy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_joints = pick_joints

        if nearest_joints is not None:
            return {
                joint_id: max(0.0, min(1.0, nearest_joints[joint_id]))
                for joint_id in home_joints
                if joint_id in nearest_joints
            }
        return None

    if len(parsed) < min_samples:
        return None

    tx, ty = float(offset_xy[0]), float(offset_xy[1])
    nearest_joints: dict[int, float] | None = None
    nearest_dist = float("inf")
    for (ox, oy), pick_joints, _label in parsed:
        dist = math.hypot(tx - ox, ty - oy)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_joints = pick_joints

    if nearest_joints is not None and nearest_dist <= snap_distance_mm:
        return {
            joint_id: max(0.0, min(1.0, nearest_joints[joint_id]))
            for joint_id in home_joints
            if joint_id in nearest_joints
        }

    weights = [
        1.0 / (math.hypot(tx - ox, ty - oy) ** power + 1e-3) for (ox, oy), _, _ in parsed
    ]
    targets: dict[int, float] = {}
    for joint_id, home_pos in home_joints.items():
        contributing = [
            (weight, pick_joints[joint_id])
            for weight, (_, pick_joints, _) in zip(weights, parsed)
            if joint_id in pick_joints
        ]
        if not contributing:
            targets[joint_id] = home_pos
            continue
        blended = sum(weight * value for weight, value in contributing) / sum(
            weight for weight, _ in contributing
        )
        targets[joint_id] = max(0.0, min(1.0, blended))
    return targets


def calibration_scales_for_arm(
    arm_type: str,
    units: str,
) -> dict[int, dict[str, float]] | None:
    calibration = load_pick_calibration()
    if calibration is None:
        return None
    if calibration.get("arm_type") != arm_type:
        return None
    if str(calibration.get("units") or "mm") != units:
        return None
    raw = calibration.get("pick_scales") or {}
    parsed: dict[int, dict[str, float]] = {}
    for key, values in raw.items():
        parsed[int(key)] = {
            "dx": float(values.get("dx", 0.0)),
            "dy": float(values.get("dy", 0.0)),
        }
    return parsed or None
