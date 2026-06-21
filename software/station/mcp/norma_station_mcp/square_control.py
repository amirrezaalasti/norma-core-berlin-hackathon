from __future__ import annotations

import asyncio
import math
import os
from typing import Any

from .pick_calibration import (
    _pick_samples_from_calibration,
    board_xy_to_offset_mm,
    load_pick_calibration,
)
from .pick_control import (
    HOME_MOTION_TIMEOUT_S,
    HOME_STABLE_READS,
    HOME_TOLERANCE_STEPS,
    MOTION_TIMEOUT_S,
    PLACE_DWELL_S,
    POSE_TOLERANCE,
    STABLE_READS,
    _ensure_gripper_open_before_close,
    _home_joint_dict,
    _home_motor_steps,
    _move_arm_exact,
    _prepare_session,
    _require_home_pose,
    _wait_arm_exact,
    close_gripper_for_pick,
    go_home,
    lift_object,
    load_home_pose,
    open_gripper_for_place,
)

DEFAULT_GRID_COLS = 5
DEFAULT_GRID_ROWS = 3


def grid_dimensions() -> tuple[int, int]:
    cols = max(1, int(os.environ.get("NORMA_BOARD_GRID_COLS", str(DEFAULT_GRID_COLS))))
    rows = max(1, int(os.environ.get("NORMA_BOARD_GRID_ROWS", str(DEFAULT_GRID_ROWS))))
    return cols, rows


def square_count() -> int:
    cols, rows = grid_dimensions()
    return cols * rows


def square_board_xy(square_id: int) -> tuple[float, float]:
    cols, rows = grid_dimensions()
    if square_id < 1 or square_id > cols * rows:
        raise ValueError(f"square_id must be between 1 and {cols * rows}, got {square_id}")
    row = (square_id - 1) // cols
    col = (square_id - 1) % cols
    return (col + 0.5) / cols, (row + 0.5) / rows


def square_info(square_id: int) -> dict[str, Any]:
    cols, rows = grid_dimensions()
    row = (square_id - 1) // cols
    col = (square_id - 1) % cols
    board_xy = square_board_xy(square_id)
    return {
        "square_id": square_id,
        "square_col": col,
        "square_row": row,
        "board_xy": [round(board_xy[0], 4), round(board_xy[1], 4)],
        "grid_cols": cols,
        "grid_rows": rows,
    }


def _workspace_from_calibration(calibration: dict[str, Any]) -> dict[str, Any]:
    workspace = calibration.get("workspace")
    if workspace is None:
        raise RuntimeError(
            "Pick calibration has no workspace. Calibrate the board in the station viewer first."
        )
    return workspace


def square_offset_mm(square_id: int, calibration: dict[str, Any] | None = None) -> tuple[float, float]:
    calibration = calibration or load_pick_calibration()
    if calibration is None:
        raise RuntimeError("No pick calibration saved.")
    board_xy = square_board_xy(square_id)
    return board_xy_to_offset_mm(board_xy, _workspace_from_calibration(calibration))


def _recorded_square_sample(
    square_id: int,
    calibration: dict[str, Any],
) -> dict[str, Any] | None:
    for sample in calibration.get("calibration_samples") or []:
        if sample.get("square_id") == square_id:
            return sample
        label = str(sample.get("label") or "")
        if label == f"square_{square_id}":
            return sample
    return None


def _idw_joint_targets(
    home_joints: dict[int, float],
    offset_xy: tuple[float, float],
    calibration: dict[str, Any],
    *,
    power: float = 2.0,
) -> dict[int, float]:
    parsed = _pick_samples_from_calibration(calibration)
    if not parsed:
        raise RuntimeError("No pick calibration samples available for square interpolation.")

    tx, ty = float(offset_xy[0]), float(offset_xy[1])
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


def joint_targets_for_square(
    square_id: int,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve joint targets for a board square from recorded poses or IDW interpolation."""
    calibration = calibration or load_pick_calibration()
    if calibration is None:
        raise RuntimeError("No pick calibration saved.")

    home = load_home_pose()
    if home is None:
        home_joints = {
            int(joint_id): float(value)
            for joint_id, value in calibration["home_joint_positions"].items()
        }
    else:
        home_joints = _home_joint_dict(home)

    info = square_info(square_id)
    recorded = _recorded_square_sample(square_id, calibration)

    if recorded and recorded.get("pick_joint_positions"):
        joints = {
            int(joint_id): float(value)
            for joint_id, value in recorded["pick_joint_positions"].items()
        }
        motor_steps_raw = recorded.get("motor_steps")
        motor_steps = (
            {int(k): int(v) for k, v in motor_steps_raw.items()}
            if motor_steps_raw
            else None
        )
        source = "recorded_static"
        sample_offset = recorded.get("offset_mm")
        offset = tuple(sample_offset) if sample_offset else (0.0, 0.0)
    else:
        offset = square_offset_mm(square_id, calibration)
        joints = _idw_joint_targets(home_joints, offset, calibration)
        motor_steps = None
        source = "interpolated"
        sample_offset = None

    return {
        **info,
        "offset_mm": [round(float(offset[0]), 2), round(float(offset[1]), 2)],
        "recorded_offset_mm": sample_offset,
        "planning_source": source,
        "joint_targets": {str(joint_id): value for joint_id, value in joints.items()},
        "motor_steps": (
            {str(joint_id): steps for joint_id, steps in motor_steps.items()}
            if motor_steps
            else None
        ),
    }


def list_square_poses() -> dict[str, Any]:
    calibration = load_pick_calibration()
    if calibration is None:
        raise RuntimeError("No pick calibration saved.")

    squares = [joint_targets_for_square(square_id, calibration) for square_id in range(1, square_count() + 1)]
    return {
        "grid_cols": grid_dimensions()[0],
        "grid_rows": grid_dimensions()[1],
        "square_count": square_count(),
        "squares": squares,
        "note": (
            "Each square maps to a 5x3 board grid (default). "
            "Recorded samples are used when available; others use IDW over calibration samples. "
            "Pick uses a partial grasp (default 0.42, set NORMA_GRIPPER_PICK_POSITION). "
            "Place opens the gripper only after reaching the square."
        ),
    }


async def go_to_square(
    session: Any,
    square_id: int,
    *,
    pick: bool = True,
    start_from_home: bool = True,
    lift_after: bool = False,
    return_home: bool = False,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    """Move to a board square and optionally pick (close gripper) at the calibrated pose."""
    home = _require_home_pose()
    home_joints = _home_joint_dict(home)
    pose = joint_targets_for_square(square_id)
    square_joints = {int(k): float(v) for k, v in pose["joint_targets"].items()}
    motor_steps_raw = pose.get("motor_steps")
    square_steps = (
        {int(k): int(v) for k, v in motor_steps_raw.items()} if motor_steps_raw else None
    )

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
            tolerance=0.02,
            tolerance_steps=HOME_TOLERANCE_STEPS,
            stable_reads=HOME_STABLE_READS,
            timeout_s=HOME_MOTION_TIMEOUT_S,
            bus_serial=bus_serial,
        )

    if pick:
        await session.open_gripper(bus_serial)
        await asyncio.sleep(0.3)

    await _move_arm_exact(
        session,
        joint_positions=square_joints,
        motor_steps=square_steps,
        bus_serial=bus_serial,
    )
    settled = await _wait_arm_exact(
        session,
        joint_positions=square_joints,
        motor_steps=square_steps,
        tolerance=POSE_TOLERANCE,
        tolerance_steps=HOME_TOLERANCE_STEPS,
        stable_reads=HOME_STABLE_READS,
        timeout_s=HOME_MOTION_TIMEOUT_S if square_steps else MOTION_TIMEOUT_S,
        bus_serial=bus_serial,
    )

    result: dict[str, Any] = {
        "action": "go_to_square",
        "square_id": square_id,
        "pick": pick,
        "planning_source": pose["planning_source"],
        "board_xy": pose["board_xy"],
        "offset_mm": pose["offset_mm"],
        "joint_targets": pose["joint_targets"],
        "motor_steps": pose.get("motor_steps"),
        "motion_settled": settled,
    }

    if pick:
        await _ensure_gripper_open_before_close(session, bus_serial)
        gripper_result = await close_gripper_for_pick(session, bus_serial)
        result["gripper_grasp"] = gripper_result

    if lift_after:
        result["lifted_home"] = await lift_object(session, bus_serial=bus_serial)
    elif return_home:
        result["returned_home"] = await go_home(
            session,
            bus_serial=bus_serial,
            open_gripper=not pick,
            wait=True,
        )

    return result


def _motion_close_enough(
    settled: dict[str, Any],
    *,
    used_motor_steps: bool,
) -> bool:
    if settled.get("reached"):
        return True
    if used_motor_steps:
        return int(settled.get("max_error_steps") or 999) <= HOME_TOLERANCE_STEPS
    max_error = settled.get("max_error")
    if max_error is None:
        return False
    return float(max_error) <= POSE_TOLERANCE * 2


async def transfer_object(
    session: Any,
    from_square: int,
    to_square: int,
    *,
    return_home: bool = True,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    """Pick at from_square, place at to_square (gripper stays closed between moves)."""
    if from_square == to_square:
        raise ValueError("from_square and to_square must be different")

    pick_result = await go_to_square(
        session,
        from_square,
        pick=True,
        start_from_home=True,
        lift_after=False,
        bus_serial=bus_serial,
    )
    place_result = await place_at_square(
        session,
        to_square,
        return_home=return_home,
        bus_serial=bus_serial,
    )
    return {
        "action": "transfer_object",
        "from_square": from_square,
        "to_square": to_square,
        "pick": pick_result,
        "place": place_result,
    }


async def place_at_square(
    session: Any,
    square_id: int,
    *,
    return_home: bool = False,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    """Place a held object at a board square: move with gripper closed, open last."""
    pose = joint_targets_for_square(square_id)
    square_joints = {int(k): float(v) for k, v in pose["joint_targets"].items()}
    motor_steps_raw = pose.get("motor_steps")
    square_steps = (
        {int(k): int(v) for k, v in motor_steps_raw.items()} if motor_steps_raw else None
    )

    await _prepare_session(session, bus_serial)
    await _move_arm_exact(
        session,
        joint_positions=square_joints,
        motor_steps=square_steps,
        bus_serial=bus_serial,
    )
    settled = await _wait_arm_exact(
        session,
        joint_positions=square_joints,
        motor_steps=square_steps,
        tolerance=POSE_TOLERANCE,
        tolerance_steps=HOME_TOLERANCE_STEPS,
        stable_reads=HOME_STABLE_READS,
        timeout_s=HOME_MOTION_TIMEOUT_S if square_steps else MOTION_TIMEOUT_S,
        bus_serial=bus_serial,
    )

    if not _motion_close_enough(settled, used_motor_steps=square_steps is not None):
        err = settled.get("max_error_steps") or settled.get("max_error")
        raise RuntimeError(
            f"Arm did not settle at square {square_id} before place "
            f"(error={err}). Gripper was not opened."
        )

    gripper_result = await open_gripper_for_place(session, bus_serial)

    result: dict[str, Any] = {
        "action": "place_at_square",
        "square_id": square_id,
        "planning_source": pose["planning_source"],
        "board_xy": pose["board_xy"],
        "offset_mm": pose["offset_mm"],
        "joint_targets": pose["joint_targets"],
        "motion_settled": settled,
        "gripper_opened": True,
        "gripper_release": gripper_result,
        "note": (
            "Arm must fully settle at the square before the gripper opens. "
            f"Dwell {PLACE_DWELL_S}s at target, then open."
        ),
    }

    if return_home:
        result["returned_home"] = await go_home(
            session,
            bus_serial=bus_serial,
            open_gripper=True,
            wait=True,
        )

    return result
