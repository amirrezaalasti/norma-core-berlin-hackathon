from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT
from .pick_control import _prepare_session

DIRECTION_MOTION_TIMEOUT_S = 6.0
DIRECTION_TOLERANCE_STEPS = 15
DIRECTION_STABLE_READS = 2

DIRECTION_NUDGE_PATH = Path(
    os.environ.get(
        "NORMA_DIRECTION_NUDGE_PATH",
        str(REPO_ROOT / ".norma" / "direction_nudge.json"),
    )
)

DIRECTION_ALIASES: dict[str, str] = {
    "up": "up",
    "raise": "up",
    "lift": "up",
    "higher": "up",
    "down": "down",
    "lower": "down",
    "left": "left",
    "right": "right",
}

# Built-in fallback for ElRobot when the JSON file is missing.
DEFAULT_ELROBOT_NUDGES: dict[str, dict[str, float]] = {
    "up": {
        "1": -0.0019,
        "2": 0.0032,
        "3": -0.0234,
        "4": -0.1626,
        "5": -0.001,
        "6": -0.0246,
        "7": -0.0003,
    },
    "down": {
        "1": -0.0039,
        "2": 0.2007,
        "3": 0.009,
        "4": 0.09,
        "5": -0.0629,
        "6": 0.2879,
        "7": -0.0006,
    },
    "right": {
        "1": 0.1759,
        "2": 0.2835,
        "3": 0.0013,
        "4": -0.0595,
        "5": -0.1006,
        "6": 0.2912,
        "7": -0.0008,
    },
    "left": {
        "1": -0.3769,
        "2": 0.0169,
        "3": -0.0017,
        "4": 0.0135,
        "5": -0.1006,
        "6": 0.2903,
        "7": -0.0003,
    },
}


def normalize_direction(direction: str) -> str:
    key = direction.strip().lower().replace(" ", "_")
    if key.startswith("go_"):
        key = key[3:]
    if key not in DIRECTION_ALIASES:
        valid = ", ".join(sorted({v for v in DIRECTION_ALIASES.values()}))
        raise ValueError(f"Unknown direction '{direction}'. Use one of: {valid}")
    return DIRECTION_ALIASES[key]


def load_direction_nudge() -> dict[str, Any] | None:
    if not DIRECTION_NUDGE_PATH.is_file():
        return None
    return json.loads(DIRECTION_NUDGE_PATH.read_text())


def direction_deltas_for_arm(arm_type: str) -> dict[str, dict[int, float]]:
    payload = load_direction_nudge()
    if payload is not None and payload.get("arm_type") == arm_type:
        directions = payload.get("directions") or {}
        parsed: dict[str, dict[int, float]] = {}
        for name, entry in directions.items():
            raw = entry.get("joint_deltas") or {}
            parsed[name] = {int(j): float(v) for j, v in raw.items()}
        if parsed:
            return parsed

    if arm_type == "elrobot":
        return {
            name: {int(j): float(v) for j, v in deltas.items()}
            for name, deltas in DEFAULT_ELROBOT_NUDGES.items()
        }

    raise RuntimeError(
        f"No direction nudge calibration for arm type '{arm_type}'. "
        f"Add {DIRECTION_NUDGE_PATH} or use move_joint / move_arm_pose."
    )


def _current_joint_dict(arm_state: dict[str, Any]) -> dict[int, float]:
    return {
        int(joint["motor_id"]): float(joint["present_position_normalized"])
        for joint in arm_state.get("joints") or []
    }


def _current_joint_steps(arm_state: dict[str, Any]) -> dict[int, int]:
    return {
        int(joint["motor_id"]): int(joint["present_position"])
        for joint in arm_state.get("joints") or []
    }


def _joint_ranges(arm_state: dict[str, Any]) -> dict[int, tuple[int, int]]:
    return {
        int(joint["motor_id"]): (int(joint["range_min"]), int(joint["range_max"]))
        for joint in arm_state.get("joints") or []
    }


def joint_step_targets_for_direction(
    current_steps: dict[int, int],
    ranges: dict[int, tuple[int, int]],
    direction: str,
    *,
    arm_type: str,
    amount: float = 1.0,
) -> dict[int, int]:
    normalized = normalize_direction(direction)
    nudges = direction_deltas_for_arm(arm_type)
    deltas = nudges[normalized]

    targets: dict[int, int] = {}
    for joint_id, steps in current_steps.items():
        delta_norm = float(deltas.get(joint_id, 0.0)) * amount
        if abs(delta_norm) < 1e-5:
            targets[joint_id] = steps
            continue
        range_min, range_max = ranges[joint_id]
        span = range_max - range_min
        targets[joint_id] = steps + int(round(delta_norm * span))
    return targets


def joint_targets_for_direction(
    current_joints: dict[int, float],
    direction: str,
    *,
    arm_type: str,
    amount: float = 1.0,
) -> dict[int, float]:
    normalized = normalize_direction(direction)
    nudges = direction_deltas_for_arm(arm_type)
    deltas = nudges[normalized]

    targets: dict[int, float] = {}
    for joint_id, home_pos in current_joints.items():
        delta = float(deltas.get(joint_id, 0.0)) * amount
        if abs(delta) < 1e-5:
            targets[joint_id] = home_pos
            continue
        targets[joint_id] = max(0.0, min(1.0, home_pos + delta))
    return targets


async def move_direction(
    session: Any,
    direction: str,
    *,
    amount: float = 1.0,
    bus_serial: str = "auto",
    wait: bool = False,
) -> dict[str, Any]:
    """Move the arm one calibrated teleop nudge in up/down/left/right."""
    if amount <= 0:
        raise ValueError("amount must be positive")

    normalized = normalize_direction(direction)
    await _prepare_session(session, bus_serial)
    arm_state = session.get_arm_state(bus_serial)
    arm_type = str(arm_state.get("arm_type") or "unknown")
    current = _current_joint_dict(arm_state)
    current_steps = _current_joint_steps(arm_state)
    ranges = _joint_ranges(arm_state)
    targets = joint_step_targets_for_direction(
        current_steps,
        ranges,
        normalized,
        arm_type=arm_type,
        amount=amount,
    )

    nudges = direction_deltas_for_arm(arm_type)
    applied_deltas = {
        joint_id: targets[joint_id] - current_steps[joint_id]
        for joint_id in targets
        if targets[joint_id] != current_steps[joint_id]
    }
    primary = sorted(
        applied_deltas,
        key=lambda joint_id: abs(applied_deltas[joint_id]),
        reverse=True,
    )[:3]

    await session.move_motors_steps(targets, bus_serial)

    result: dict[str, Any] = {
        "action": "move_direction",
        "direction": normalized,
        "amount": amount,
        "arm_type": arm_type,
        "motor_steps": targets,
        "joint_targets": {
            joint_id: round(current[joint_id], 4) for joint_id in current
        },
        "applied_step_deltas": applied_deltas,
        "primary_joints": primary,
        "note": (
            "Direction moves use teleop-calibrated joint deltas from the current pose "
            "in motor steps (works outside normalized 0-1). Use amount=2.0 for a double nudge."
        ),
    }
    if wait:
        result["motion_settled"] = await session.wait_for_arm_steps(
            targets,
            tolerance_steps=DIRECTION_TOLERANCE_STEPS,
            stable_reads=DIRECTION_STABLE_READS,
            timeout_s=DIRECTION_MOTION_TIMEOUT_S,
            bus_serial=bus_serial,
        )
    else:
        result["note"] += " Returns immediately; arm keeps moving (wait=true to block until settled)."
    return result
