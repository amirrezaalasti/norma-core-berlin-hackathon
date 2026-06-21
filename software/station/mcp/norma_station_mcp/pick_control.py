from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT

HOME_POSE_PATH = Path(
    os.environ.get("NORMA_HOME_POSE_PATH", str(REPO_ROOT / ".norma" / "home_pose.json"))
)

DEFAULT_PICK_SCALES: dict[str, dict[int, dict[str, float]]] = {
    "so101": {
        1: {"dx": -0.00055, "dy": 0.0},
        2: {"dx": 0.0, "dy": 0.00105},
        3: {"dx": 0.0, "dy": 0.00085},
        4: {"dx": -0.00025, "dy": 0.00045},
        5: {"dx": 0.00018, "dy": 0.00022},
    },
    "elrobot": {
        1: {"dx": -0.00045, "dy": 0.0},
        2: {"dx": 0.0, "dy": 0.00095},
        3: {"dx": 0.0, "dy": 0.00075},
        4: {"dx": -0.0002, "dy": 0.0004},
        5: {"dx": 0.00015, "dy": 0.0002},
        6: {"dx": 0.0, "dy": 0.00015},
        7: {"dx": 0.0001, "dy": 0.0001},
    },
}


DEFAULT_PICK_SCALES_MM: dict[str, dict[int, dict[str, float]]] = {
    "so101": {
        1: {"dx": -0.0008, "dy": 0.0},
        2: {"dx": 0.0, "dy": 0.0015},
        3: {"dx": 0.0, "dy": 0.0012},
        4: {"dx": -0.00035, "dy": 0.00065},
        5: {"dx": 0.00025, "dy": 0.00032},
    },
    "elrobot": {
        1: {"dx": -0.000035, "dy": 0.000034},
        2: {"dx": 0.001387, "dy": -0.003754},
        3: {"dx": -0.000100, "dy": -0.000493},
        4: {"dx": -0.001052, "dy": -0.002166},
        5: {"dx": 0.000665, "dy": 0.004356},
        6: {"dx": -0.000885, "dy": -0.002501},
        7: {"dx": 0.000001, "dy": 0.000020},
    },
}


def _load_pick_scales(arm_type: str, units: str = "px") -> dict[int, dict[str, float]]:
    raw = os.environ.get("NORMA_PICK_SCALES")
    if raw:
        parsed = json.loads(raw)
        if arm_type in parsed:
            return {int(k): v for k, v in parsed[arm_type].items()}

    from .pick_calibration import calibration_scales_for_arm

    calibrated = calibration_scales_for_arm(arm_type, units=units)
    if calibrated is not None:
        return calibrated

    if units == "mm":
        return DEFAULT_PICK_SCALES_MM.get(arm_type, DEFAULT_PICK_SCALES_MM["so101"])
    return DEFAULT_PICK_SCALES.get(arm_type, DEFAULT_PICK_SCALES["so101"])


def load_home_pose() -> dict[str, Any] | None:
    if not HOME_POSE_PATH.is_file():
        return None
    return json.loads(HOME_POSE_PATH.read_text())


def save_home_pose(payload: dict[str, Any]) -> dict[str, Any]:
    HOME_POSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOME_POSE_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def home_pose_from_arm_state(arm_state: dict[str, Any]) -> dict[str, Any]:
    joint_positions: dict[int, float] = {}
    for joint in arm_state.get("joints", []):
        normalized = joint.get("present_position_normalized")
        if normalized is not None:
            joint_positions[joint["motor_id"]] = float(normalized)

    gripper = arm_state.get("gripper") or {}
    gripper_position = gripper.get("present_position_normalized")

    return {
        "bus_serial": arm_state["bus_serial"],
        "arm_type": arm_state["arm_type"],
        "joint_positions": joint_positions,
        "gripper_position": gripper_position,
        "note": (
            "Saved while arm is at the initialized pose. Pair with manual calibration: "
            "Set 4 board points and the gripper tip in the station viewer."
        ),
    }


def offset_to_joint_targets(
    home: dict[str, Any],
    offset_xy: tuple[float, float],
    arm_type: str,
    units: str = "px",
) -> dict[int, float]:
    from .pick_calibration import (
        calibration_scales_for_arm,
        joint_targets_from_calibration_samples,
        load_pick_calibration,
    )

    calibration = load_pick_calibration()
    if (
        calibration is not None
        and calibration.get("arm_type") == arm_type
        and str(calibration.get("units") or "mm") == units
    ):
        home_joints = {int(k): float(v) for k, v in home["joint_positions"].items()}
        sampled = joint_targets_from_calibration_samples(
            home_joints,
            offset_xy,
            calibration,
        )
        if sampled is not None:
            return sampled

    scales = calibration_scales_for_arm(arm_type, units=units)
    if scales is not None:
        return _joint_targets_from_scales(home, offset_xy, scales)

    if (
        calibration is not None
        and calibration.get("arm_type") == arm_type
        and str(calibration.get("units") or "mm") == units
        and calibration.get("reference_offset_mm")
        and calibration.get("reference_pick_joint_positions")
    ):
        return _joint_targets_from_reference(home, offset_xy, calibration)

    dx, dy = offset_xy
    scales = _load_pick_scales(arm_type, units=units)
    home_joints = {int(k): float(v) for k, v in home["joint_positions"].items()}
    targets: dict[int, float] = {}

    for joint_id, home_pos in home_joints.items():
        scale = scales.get(joint_id, {"dx": 0.0, "dy": 0.0})
        delta = scale["dx"] * dx + scale["dy"] * dy
        targets[joint_id] = max(0.0, min(1.0, home_pos + delta))

    return targets


def _joint_targets_from_scales(
    home: dict[str, Any],
    offset_xy: tuple[float, float],
    scales: dict[int, dict[str, float]],
) -> dict[int, float]:
    dx, dy = float(offset_xy[0]), float(offset_xy[1])
    home_joints = {int(k): float(v) for k, v in home["joint_positions"].items()}
    targets: dict[int, float] = {}
    for joint_id, home_pos in home_joints.items():
        scale = scales.get(joint_id, {"dx": 0.0, "dy": 0.0})
        delta = scale["dx"] * dx + scale["dy"] * dy
        targets[joint_id] = max(0.0, min(1.0, home_pos + delta))
    return targets


def _joint_targets_from_reference(
    home: dict[str, Any],
    offset_xy: tuple[float, float],
    calibration: dict[str, Any],
) -> dict[int, float]:
    ref_offset = calibration["reference_offset_mm"]
    ref_pick = calibration["reference_pick_joint_positions"]
    home_joints = {int(k): float(v) for k, v in home["joint_positions"].items()}

    ref_x, ref_y = float(ref_offset[0]), float(ref_offset[1])
    cur_x, cur_y = float(offset_xy[0]), float(offset_xy[1])
    denom = ref_x * ref_x + ref_y * ref_y
    if denom <= 1e-6:
        alpha = 1.0
    else:
        alpha = (cur_x * ref_x + cur_y * ref_y) / denom

    targets: dict[int, float] = {}
    for joint_id, home_pos in home_joints.items():
        ref_key = str(joint_id)
        if ref_key not in ref_pick:
            targets[joint_id] = home_pos
            continue
        ref_pos = float(ref_pick[ref_key])
        delta = ref_pos - home_pos
        targets[joint_id] = max(0.0, min(1.0, home_pos + alpha * delta))
    return targets


def _detection_units(workspace: dict[str, Any] | None) -> str:
    if workspace is None:
        return "px"
    units = str(workspace.get("units") or "px")
    if units == "mm":
        return "mm"
    return "px"


def _load_manual_workspace_dict() -> dict[str, Any] | None:
    path = REPO_ROOT / ".norma" / "manual_workspace.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("calibration_source") != "manual" or not data.get("corners_xy"):
        return None
    return data


def _manual_workspace_ready(workspace: dict[str, Any] | None) -> bool:
    return bool(
        workspace is not None
        and workspace.get("calibration_source") == "manual"
        and workspace.get("gripper_tip_set")
        and workspace.get("corners_xy")
    )


def _apply_manual_workspace_to_detections(
    detections: list[dict[str, Any]],
    workspace_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    _ensure_vision_importable()
    from norma_vision.manual_workspace_store import _workspace_from_dict
    from norma_vision.workspace import enrich_detections_with_workspace

    workspace = _workspace_from_dict(workspace_dict)
    if workspace is None:
        return detections
    return enrich_detections_with_workspace(detections, workspace)


def _ensure_vision_importable() -> None:
    import sys

    vision_path = str(REPO_ROOT / "software" / "station" / "vision")
    if vision_path not in sys.path:
        sys.path.insert(0, vision_path)


def _workspace_ready_for_pick(workspace: dict[str, Any] | None) -> bool:
    if _manual_workspace_ready(workspace):
        return True
    if workspace is None:
        return False
    source = workspace.get("calibration_source")
    if source == "manual":
        return bool(workspace.get("gripper_tip_set"))
    return source in ("apriltag", "markers", "blue_dots", "gripper", "camera")


def pick_target_from_detection(
    home: dict[str, Any],
    detection: dict[str, Any],
    arm_type: str,
    units: str = "px",
) -> dict[str, Any]:
    offset = detection.get("offset_xy")
    distance = detection.get("distance")
    if offset is None or distance is None:
        raise ValueError("Detection is missing offset_xy/distance relative to gripper origin")

    joint_targets = offset_to_joint_targets(
        home,
        (float(offset[0]), float(offset[1])),
        arm_type,
        units=units,
    )
    from .pick_calibration import calibration_scales_for_arm, load_pick_calibration

    calibration = load_pick_calibration()
    planning_mode = (
        "multi_point_2d"
        if calibration is not None
        and len(calibration.get("calibration_samples") or []) >= 2
        else (
            "2d_scales"
            if calibration_scales_for_arm(arm_type, units=units) is not None
            else "default_scales"
        )
    )

    gripper_pick = None
    if calibration is not None:
        if calibration.get("planning_mode") == "static_hardcoded":
            planning_mode = "static_hardcoded"
            from .pick_calibration import nearest_sample_from_calibration
            nearest = nearest_sample_from_calibration((float(offset[0]), float(offset[1])), calibration)
            if nearest is not None and "gripper_pick" in nearest:
                gripper_pick = nearest["gripper_pick"]

    res = {
        "class_name": detection.get("class_name"),
        "confidence": detection.get("confidence"),
        "offset_xy": offset,
        "distance": distance,
        "units": units,
        "planning_mode": planning_mode,
        "joint_targets_from_home": joint_targets,
        "home_joint_positions": home["joint_positions"],
    }
    if gripper_pick is not None:
        res["gripper_pick"] = gripper_pick
    return res


async def save_pick_reference(
    session: Any,
    board_x: float,
    board_y: float,
    bus_serial: str = "auto",
    workspace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .pick_calibration import (
        board_xy_to_offset_mm,
        load_pick_calibration,
        pick_calibration_from_poses,
        save_pick_calibration,
    )

    home = load_home_pose()
    if home is None:
        raise RuntimeError("Call save_home_pose first while the arm is at the home position.")

    await session.ensure_connected()
    await session.wait_for_inference()
    arm_state = session.get_arm_state(bus_serial)

    if workspace is None:
        workspace_path = REPO_ROOT / ".norma" / "manual_workspace.json"
        if workspace_path.is_file():
            workspace = json.loads(workspace_path.read_text())
    if workspace is None:
        raise RuntimeError(
            "Manual workspace is required. Calibrate in the station viewer or pass workspace JSON."
        )

    offset_mm = board_xy_to_offset_mm((board_x, board_y), workspace)
    home_joints = {int(k): float(v) for k, v in home["joint_positions"].items()}
    pick_joints = {
        int(joint["motor_id"]): float(joint["present_position_normalized"])
        for joint in arm_state.get("joints", [])
        if joint.get("present_position_normalized") is not None
    }
    gripper = arm_state.get("gripper") or {}

    existing = load_pick_calibration()
    samples = list((existing or {}).get("calibration_samples") or [])
    samples.append(
        {
            "label": f"board_{board_x:.2f}_{board_y:.2f}",
            "board_xy": [round(board_x, 4), round(board_y, 4)],
            "offset_mm": [round(offset_mm[0], 2), round(offset_mm[1], 2)],
            "pick_joint_positions": {
                str(joint_id): round(value, 4) for joint_id, value in pick_joints.items()
            },
        }
    )

    payload = pick_calibration_from_poses(
        bus_serial=arm_state["bus_serial"],
        arm_type=arm_state["arm_type"],
        home_joints=home_joints,
        pick_joints=pick_joints,
        gripper_home=home.get("gripper_position"),
        gripper_pick=gripper.get("present_position_normalized"),
        offset_mm=offset_mm,
        board_xy=(board_x, board_y),
        workspace=workspace,
        units=str(workspace.get("units") or "mm"),
        calibration_samples=samples,
    )
    save_pick_calibration(payload)
    return payload


async def pick_nearest_object(
    session: Any,
    bus_serial: str = "auto",
    settle_s: float = 1.5,
    return_home: bool = True,
) -> dict[str, Any]:
    from .vision_bridge import detect_workspace_objects

    home = load_home_pose()
    if home is None:
        raise RuntimeError(
            "No home pose saved. Move the arm to the initialized pose and call save_home_pose first."
        )

    manual_workspace = _load_manual_workspace_dict()
    if not _manual_workspace_ready(manual_workspace):
        raise RuntimeError(
            "Manual workspace calibration is required. In the station viewer, click "
            "'Set 4 points' for the board corners, then 'Set gripper tip' while the arm "
            "is at the home pose."
        )

    await session.ensure_connected()
    vision = await detect_workspace_objects(camera_index=0)
    workspace = manual_workspace
    raw_detections = vision.get("detections", [])
    detections = _apply_manual_workspace_to_detections(raw_detections, manual_workspace)
    detections = [
        item
        for item in detections
        if item.get("offset_xy") is not None and item.get("distance") is not None
    ]
    if not detections:
        raise RuntimeError("No objects detected with gripper-relative offsets")

    target_detection = min(detections, key=lambda item: float(item["distance"]))
    units = _detection_units(workspace)
    pick_plan = pick_target_from_detection(
        home,
        target_detection,
        home["arm_type"],
        units=units,
    )

    await session.wait_for_inference()
    await session.enable_arm_torque(bus_serial)
    await session.open_gripper(bus_serial)
    await session.move_arm_pose(pick_plan["joint_targets_from_home"], bus_serial)
    await asyncio.sleep(settle_s)

    gripper_pick = pick_plan.get("gripper_pick")
    if gripper_pick is None:
        from .pick_calibration import load_pick_calibration
        calibration = load_pick_calibration()
        gripper_pick = (calibration or {}).get("gripper_pick")

    if gripper_pick is not None:
        await session.set_gripper(float(gripper_pick), bus_serial)
    else:
        await session.close_gripper(bus_serial)

    result: dict[str, Any] = {
        "picked": target_detection,
        "pick_plan": pick_plan,
        "workspace": workspace,
        "gripper_tip": vision.get("gripper_tip"),
        "camera_calibration": vision.get("camera_calibration"),
    }

    if return_home:
        await asyncio.sleep(0.5)
        home_joints = {int(k): float(v) for k, v in home["joint_positions"].items()}
        await session.move_arm_pose(home_joints, bus_serial)
        gripper_home = home.get("gripper_position")
        if gripper_home is not None:
            await session.set_gripper(float(gripper_home), bus_serial)
        result["returned_home"] = True

    return result
