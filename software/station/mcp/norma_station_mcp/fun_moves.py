from __future__ import annotations

import asyncio
from typing import Any

from .pick_control import (
    MOTION_TIMEOUT_S,
    _home_joint_dict,
    _prepare_session,
    fully_close_gripper,
    fully_open_gripper,
    go_home,
    load_home_pose,
)

# Snappy timing — fun moves trade precision for energy.
GRIPPER_FLAP_INTERVAL_S = 0.06
SAY_HI_CYCLES = 2
SAY_HI_BETWEEN_CYCLES_S = 0.08
DANCE_POSE_DWELL_S = 0.1
ACK_QUICK_PAUSE_S = 0.1
FUN_POSE_TOLERANCE = 0.025
FUN_STABLE_READS = 2


def _clamp_joint_targets(joints: dict[int, float]) -> dict[int, float]:
    return {joint_id: max(0.0, min(1.0, value)) for joint_id, value in joints.items()}


def _offset_joints(base: dict[int, float], deltas: dict[int, float]) -> dict[int, float]:
    targets = dict(base)
    for joint_id, delta in deltas.items():
        if joint_id in targets:
            targets[joint_id] = targets[joint_id] + delta
    return _clamp_joint_targets(targets)


async def _base_joints(session: Any, bus_serial: str = "auto") -> dict[int, float]:
    home = load_home_pose()
    if home is not None:
        return _home_joint_dict(home)
    arm_state = session.get_arm_state(bus_serial)
    return {
        int(joint["motor_id"]): float(joint["present_position_normalized"])
        for joint in arm_state.get("joints") or []
    }


async def _move_and_wait(
    session: Any,
    joint_targets: dict[int, float],
    *,
    bus_serial: str = "auto",
    dwell_s: float = 0.0,
) -> dict[str, Any]:
    await session.move_arm_pose(joint_targets, bus_serial)
    settled = await session.wait_for_arm_pose(
        joint_targets,
        tolerance=FUN_POSE_TOLERANCE,
        stable_reads=FUN_STABLE_READS,
        timeout_s=MOTION_TIMEOUT_S,
        bus_serial=bus_serial,
    )
    if dwell_s > 0:
        await asyncio.sleep(dwell_s)
    return settled


async def gripper_wave(
    session: Any,
    *,
    cycles: int = 4,
    interval_s: float = GRIPPER_FLAP_INTERVAL_S,
    end_open: bool = True,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    """Rapidly open and close the gripper like a friendly wave."""
    if cycles < 1:
        raise ValueError("cycles must be at least 1")

    await _prepare_session(session, bus_serial)
    steps: list[dict[str, Any]] = []

    for _ in range(cycles):
        steps.append(await session.open_gripper(bus_serial))
        await asyncio.sleep(interval_s)
        steps.append(await session.close_gripper(bus_serial))
        await asyncio.sleep(interval_s)

    if end_open:
        steps.append(await session.open_gripper(bus_serial))

    return {
        "action": "gripper_wave",
        "cycles": cycles,
        "interval_s": interval_s,
        "end_open": end_open,
        "steps": len(steps),
    }



_DANCE_POSES = (
    {1: -0.12, 2: 0.10, 3: -0.06, 5: -0.14, 6: 0.12},
    {1: 0.12, 2: 0.10, 3: -0.06, 5: 0.14, 6: -0.12},
    {2: -0.08, 3: 0.12, 4: 0.08, 6: 0.14, 7: 0.06},
    {2: 0.08, 3: -0.10, 4: -0.06, 6: -0.12, 7: -0.06},
)

# Small head nudge from the current pose — one quick move out and back.
_ACK_NUDGE = {2: 0.05, 5: -0.05, 6: 0.06}


async def _current_joints(session: Any, bus_serial: str = "auto") -> dict[int, float]:
    arm_state = session.get_arm_state(bus_serial)
    return {
        int(joint["motor_id"]): float(joint["present_position_normalized"])
        for joint in arm_state.get("joints") or []
    }


async def acknowledge(
    session: Any,
    *,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    """Quick head nudge out and back from the current pose — 'I heard you'."""
    await _prepare_session(session, bus_serial)
    start = await _current_joints(session, bus_serial)
    nudge = _offset_joints(start, _ACK_NUDGE)

    await session.move_arm_pose(nudge, bus_serial)
    await asyncio.sleep(ACK_QUICK_PAUSE_S)
    await session.move_arm_pose(start, bus_serial)
    await asyncio.sleep(ACK_QUICK_PAUSE_S * 0.5)

    return {
        "action": "acknowledge",
        "start_pose": {str(k): round(v, 4) for k, v in start.items()},
        "nudge_pose": {str(k): round(v, 4) for k, v in nudge.items()},
        "note": "Single quick nudge out and back; does not wait for full settle.",
    }


async def say_hi(
    session: Any,
    *,
    cycles: int = SAY_HI_CYCLES,
    end_open: bool = True,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    """Say hi: fully open and close the gripper twice, waiting for each move to finish."""
    if cycles < 1:
        raise ValueError("cycles must be at least 1")

    await _prepare_session(session, bus_serial)
    steps: list[dict[str, Any]] = []

    for cycle in range(cycles):
        steps.append(await fully_open_gripper(session, bus_serial))
        steps.append(await fully_close_gripper(session, bus_serial))
        if cycle + 1 < cycles:
            await asyncio.sleep(SAY_HI_BETWEEN_CYCLES_S)

    if end_open:
        steps.append(await fully_open_gripper(session, bus_serial))

    return {
        "action": "say_hi",
        "cycles": cycles,
        "end_open": end_open,
        "gripper_range": "full (0.0 closed, 1.0 open)",
        "steps": steps,
        "note": "Waits for gripper to fully open/close before each step.",
    }


async def dance(
    session: Any,
    *,
    beats: int = 4,
    flap_each_beat: bool = True,
    flaps_per_beat: int = 3,
    return_home: bool = True,
    bus_serial: str = "auto",
) -> dict[str, Any]:
    """Energetic celebratory dance: big sways with rapid gripper flaps."""
    if beats < 1:
        raise ValueError("beats must be at least 1")

    await _prepare_session(session, bus_serial)

    if load_home_pose() is not None:
        await go_home(session, bus_serial=bus_serial, open_gripper=True, wait=True)

    base = await _base_joints(session, bus_serial)
    await session.open_gripper(bus_serial)

    poses = [_offset_joints(base, deltas) for deltas in _DANCE_POSES]

    beat_log: list[dict[str, Any]] = []
    for beat in range(beats):
        pose = poses[beat % len(poses)]
        beat_log.append(
            {
                "beat": beat + 1,
                "joint_targets": pose,
                "motion_settled": await _move_and_wait(
                    session,
                    pose,
                    bus_serial=bus_serial,
                    dwell_s=DANCE_POSE_DWELL_S,
                ),
            }
        )
        if flap_each_beat:
            beat_log[-1]["gripper_flap"] = await gripper_wave(
                session,
                cycles=flaps_per_beat,
                interval_s=GRIPPER_FLAP_INTERVAL_S,
                end_open=True,
                bus_serial=bus_serial,
            )

    result: dict[str, Any] = {
        "action": "dance",
        "beats": beats,
        "flap_each_beat": flap_each_beat,
        "flaps_per_beat": flaps_per_beat,
        "beat_log": beat_log,
    }

    if return_home and load_home_pose() is not None:
        result["returned_home"] = await go_home(
            session,
            bus_serial=bus_serial,
            open_gripper=True,
            wait=True,
        )

    return result
