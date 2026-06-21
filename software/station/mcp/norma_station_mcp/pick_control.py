from __future__ import annotations

import asyncio
import json
import os
import time
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


STATIC_PICK_JOINTS: dict[int, float] = {
    1: 0.5533,
    2: 0.8377,
    3: 0.5661,
    4: 0.4003,
    5: 0.4669,
    6: 0.8688,
    7: 0.2392,
}

POSE_TOLERANCE = 0.015
HOME_TOLERANCE = 0.02
HOME_TOLERANCE_STEPS = 12
HOME_STABLE_READS = 2
HOME_MOTION_TIMEOUT_S = 12.0
STABLE_READS = 5
MOTION_TIMEOUT_S = 45.0
PLACE_DWELL_S = 0.5
GRIPPER_OPEN_TOLERANCE = 0.85
GRIPPER_CLOSED_TOLERANCE = 0.15
GRIPPER_MOTION_TIMEOUT_S = 8.0
# Partial grasp for picks: 0.0 = fully closed, 1.0 = fully open. Override with NORMA_GRIPPER_PICK_POSITION.
DEFAULT_GRIPPER_PICK_POSITION = 0.42


def gripper_pick_position() -> float:
    raw = os.environ.get("NORMA_GRIPPER_PICK_POSITION")
    if raw is None:
        return DEFAULT_GRIPPER_PICK_POSITION
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise ValueError("NORMA_GRIPPER_PICK_POSITION must be between 0.0 and 1.0")
    return value


def load_fixed_pick_joints() -> dict[int, float]:
    """Return the static pick joint pose (always the same, not vision-derived)."""
    return dict(STATIC_PICK_JOINTS)


def _require_home_pose() -> dict[str, Any]:
    home = load_home_pose()
    if home is None:
        raise RuntimeError(
            "No home pose saved. Move the arm to the initialized pose and call save_home_pose first."
        )
    return home


def _home_joint_dict(home: dict[str, Any]) -> dict[int, float]:
    return {int(k): float(v) for k, v in home["joint_positions"].items()}


def _home_motor_steps(home: dict[str, Any]) -> dict[int, int] | None:
    motor_steps = home.get("motor_steps")
    if not motor_steps:
        return None
    return {int(k): int(v) for k, v in motor_steps.items()}


async def _move_arm_exact(
    session: Any,
    *,
    joint_positions: dict[int, float],
    motor_steps: dict[int, int] | None,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    if motor_steps:
        return await session.move_motors_steps(motor_steps, bus_serial)
    return await session.move_arm_pose(joint_positions, bus_serial)


async def _wait_arm_exact(
    session: Any,
    *,
    joint_positions: dict[int, float],
    motor_steps: dict[int, int] | None,
    tolerance: float,
    stable_reads: int,
    timeout_s: float,
    tolerance_steps: int = 3,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    if motor_steps:
        return await session.wait_for_arm_steps(
            motor_steps,
            tolerance_steps=tolerance_steps,
            stable_reads=stable_reads,
            timeout_s=timeout_s,
            bus_serial=bus_serial,
        )
    return await session.wait_for_arm_pose(
        joint_positions,
        tolerance=tolerance,
        stable_reads=stable_reads,
        timeout_s=timeout_s,
        bus_serial=bus_serial,
    )


async def _prepare_session(session: Any, bus_serial: str = "auto") -> None:
    await session.ensure_connected()
    await session.wait_for_inference()
    try:
        arm_state = session.get_arm_state(bus_serial)
        joints = arm_state.get("joints") or []
        gripper = arm_state.get("gripper") or {}
        torque_on = all(j.get("torque_enabled") for j in joints) and gripper.get(
            "torque_enabled", True
        )
        if not torque_on:
            await session.enable_arm_torque(bus_serial)
    except Exception:
        pass


async def _ensure_gripper_open_before_close(session: Any, bus_serial: str = "auto") -> None:
    arm_state = session.get_arm_state(bus_serial)
    gripper = arm_state.get("gripper") or {}
    if float(gripper.get("present_position_normalized") or 0.0) < 0.85:
        await session.open_gripper(bus_serial)
        await asyncio.sleep(0.5)


async def _wait_for_gripper(
    session: Any,
    *,
    target_min: float | None = None,
    target_max: float | None = None,
    bus_serial: str = "auto",
    timeout_s: float = GRIPPER_MOTION_TIMEOUT_S,
    poll_s: float = 0.1,
) -> float:
    """Block until gripper reaches open (target_min) or closed (target_max)."""
    if (target_min is None) == (target_max is None):
        raise ValueError("Specify exactly one of target_min or target_max")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await session.wait_for_inference(timeout_s=min(2.0, max(0.1, deadline - time.monotonic())))
        arm_state = session.get_arm_state(bus_serial)
        gripper = arm_state.get("gripper") or {}
        present = float(gripper.get("present_position_normalized") or 0.0)
        if target_min is not None and present >= target_min:
            return present
        if target_max is not None and present <= target_max:
            return present
        await asyncio.sleep(poll_s)

    label = f">= {target_min}" if target_min is not None else f"<= {target_max}"
    raise RuntimeError(f"Gripper did not reach {label} within {timeout_s}s")


async def open_gripper_for_place(session: Any, bus_serial: str = "auto") -> dict[str, Any]:
    """Open gripper for placement only after a brief dwell at the target pose."""
    await asyncio.sleep(PLACE_DWELL_S)
    result = await session.open_gripper(bus_serial)
    present = await _wait_for_gripper(
        session,
        target_min=GRIPPER_OPEN_TOLERANCE,
        bus_serial=bus_serial,
    )
    result["present_position_normalized"] = present
    return result


async def fully_open_gripper(session: Any, bus_serial: str = "auto") -> dict[str, Any]:
    """Fully open the gripper and wait until it reaches the open position."""
    result = await session.open_gripper(bus_serial)
    present = await _wait_for_gripper(
        session,
        target_min=GRIPPER_OPEN_TOLERANCE,
        bus_serial=bus_serial,
        poll_s=0.05,
    )
    result["present_position_normalized"] = present
    result["gripper_state"] = "open"
    return result


async def fully_close_gripper(session: Any, bus_serial: str = "auto") -> dict[str, Any]:
    """Fully close the gripper and wait until it reaches the closed position."""
    result = await session.close_gripper(bus_serial)
    present = await _wait_for_gripper(
        session,
        target_max=GRIPPER_CLOSED_TOLERANCE,
        bus_serial=bus_serial,
        poll_s=0.05,
    )
    result["present_position_normalized"] = present
    result["gripper_state"] = "closed"
    return result


async def close_gripper_for_pick(session: Any, bus_serial: str = "auto") -> dict[str, Any]:
    """Grasp an object with a firm but not fully closed gripper."""
    position = gripper_pick_position()
    result = await session.set_gripper(position, bus_serial)
    result["gripper_pick_position"] = position
    result["gripper_state"] = "pick_grasp"
    await asyncio.sleep(1.0)
    return result


async def go_home(
    session: Any,
    bus_serial: str = "auto",
    *,
    open_gripper: bool = True,
    wait: bool = True,
) -> dict[str, Any]:
    """Move the arm to the saved home pose."""
    home = _require_home_pose()
    home_joints = _home_joint_dict(home)
    home_steps = _home_motor_steps(home)

    await _prepare_session(session, bus_serial)
    if open_gripper:
        await session.open_gripper(bus_serial)
    await _move_arm_exact(
        session,
        joint_positions=home_joints,
        motor_steps=home_steps,
        bus_serial=bus_serial,
    )

    result: dict[str, Any] = {
        "action": "go_home",
        "joint_targets": home_joints,
        "motor_steps": home_steps,
        "gripper_opened": open_gripper,
    }
    if wait:
        settled = await _wait_arm_exact(
            session,
            joint_positions=home_joints,
            motor_steps=home_steps,
            tolerance=HOME_TOLERANCE,
            tolerance_steps=HOME_TOLERANCE_STEPS,
            stable_reads=HOME_STABLE_READS,
            timeout_s=HOME_MOTION_TIMEOUT_S,
            bus_serial=bus_serial,
        )
        result["motion_settled"] = settled
    return result


async def move_to_pick_pose(
    session: Any,
    bus_serial: str = "auto",
    *,
    wait: bool = True,
) -> dict[str, Any]:
    """Move the arm to the static pick/placement pose (gripper unchanged)."""
    pick_joints = load_fixed_pick_joints()

    await _prepare_session(session, bus_serial)
    await session.move_arm_pose(pick_joints, bus_serial)

    result: dict[str, Any] = {
        "action": "move_to_pick_pose",
        "joint_targets": pick_joints,
    }
    if wait:
        result["motion_settled"] = await session.wait_for_arm_pose(
            pick_joints,
            tolerance=POSE_TOLERANCE,
            stable_reads=STABLE_READS,
            timeout_s=MOTION_TIMEOUT_S,
            bus_serial=bus_serial,
        )
    return result


async def pick_object(
    session: Any,
    bus_serial: str = "auto",
    *,
    lift_after: bool = False,
    start_from_home: bool = True,
) -> dict[str, Any]:
    """Pick using the static pose: open gripper, move down, wait, close.

    Gripper stays closed after pick unless lift_after is True (moves to home holding object).
    """
    home = _require_home_pose()
    home_joints = _home_joint_dict(home)
    pick_joints = load_fixed_pick_joints()

    await _prepare_session(session, bus_serial)

    if start_from_home:
        await session.open_gripper(bus_serial)
        home_steps = _home_motor_steps(home)
        await _move_arm_exact(
            session,
            joint_positions=home_joints,
            motor_steps=home_steps,
            bus_serial=bus_serial,
        )
        await _wait_arm_exact(
            session,
            joint_positions=home_joints,
            motor_steps=home_steps,
            tolerance=HOME_TOLERANCE,
            tolerance_steps=HOME_TOLERANCE_STEPS,
            stable_reads=HOME_STABLE_READS,
            timeout_s=HOME_MOTION_TIMEOUT_S,
            bus_serial=bus_serial,
        )

    await session.open_gripper(bus_serial)
    await asyncio.sleep(0.3)
    await session.move_arm_pose(pick_joints, bus_serial)
    settled = await session.wait_for_arm_pose(
        pick_joints,
        tolerance=POSE_TOLERANCE,
        stable_reads=STABLE_READS,
        timeout_s=MOTION_TIMEOUT_S,
        bus_serial=bus_serial,
    )
    await _ensure_gripper_open_before_close(session, bus_serial)
    gripper_result = await close_gripper_for_pick(session, bus_serial)

    result: dict[str, Any] = {
        "action": "pick_object",
        "planning_mode": "static",
        "joint_targets": pick_joints,
        "motion_settled": settled,
        "gripper_grasp": gripper_result,
    }

    if lift_after:
        result["lift"] = await lift_object(session, bus_serial=bus_serial)
    return result


async def lift_object(session: Any, bus_serial: str = "auto") -> dict[str, Any]:
    """Lift a held object by moving to home without opening the gripper."""
    home = _require_home_pose()
    home_joints = _home_joint_dict(home)
    home_steps = _home_motor_steps(home)

    await _prepare_session(session, bus_serial)
    await _move_arm_exact(
        session,
        joint_positions=home_joints,
        motor_steps=home_steps,
        bus_serial=bus_serial,
    )
    settled = await _wait_arm_exact(
        session,
        joint_positions=home_joints,
        motor_steps=home_steps,
        tolerance=HOME_TOLERANCE,
        tolerance_steps=HOME_TOLERANCE_STEPS,
        stable_reads=HOME_STABLE_READS,
        timeout_s=HOME_MOTION_TIMEOUT_S,
        bus_serial=bus_serial,
    )

    arm_state = session.get_arm_state(bus_serial)
    gripper = (arm_state.get("gripper") or {}).get("present_position_normalized")
    return {
        "action": "lift_object",
        "joint_targets": home_joints,
        "motor_steps": home_steps,
        "motion_settled": settled,
        "gripper_position": gripper,
        "note": "Gripper left closed to retain the object.",
    }


async def place_object(session: Any, bus_serial: str = "auto") -> dict[str, Any]:
    """Place a held object: move to pick pose, open gripper, return home."""
    home = _require_home_pose()
    pick_joints = load_fixed_pick_joints()

    await _prepare_session(session, bus_serial)
    await session.move_arm_pose(pick_joints, bus_serial)
    settled = await session.wait_for_arm_pose(
        pick_joints,
        tolerance=POSE_TOLERANCE,
        stable_reads=STABLE_READS,
        timeout_s=MOTION_TIMEOUT_S,
        bus_serial=bus_serial,
    )

    await open_gripper_for_place(session, bus_serial)

    home_result = await go_home(session, bus_serial=bus_serial, open_gripper=True, wait=True)
    return {
        "action": "place_object",
        "joint_targets": pick_joints,
        "motion_settled": settled,
        "gripper_opened": True,
        "returned_home": home_result,
    }


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
    close_gripper_at_pick = False
    if calibration is not None:
        if calibration.get("planning_mode") == "static_hardcoded":
            planning_mode = "static_hardcoded"
            close_gripper_at_pick = bool(calibration.get("gripper_close_at_pick", True))
            if not close_gripper_at_pick:
                gripper_pick = calibration.get("gripper_pick")

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
    if close_gripper_at_pick:
        res["close_gripper_at_pick"] = True
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
    return_home: bool = False,
) -> dict[str, Any]:
    """Pick at the static pose (alias for pick_object, no vision planning)."""
    del settle_s  # kept for API compatibility; motion uses wait_for_arm_pose

    result = await pick_object(
        session,
        bus_serial=bus_serial,
        lift_after=return_home,
        start_from_home=True,
    )
    if return_home:
        result["returned_home"] = True
    return result
