from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT
from .pick_control import _prepare_session, load_home_pose

DIRECTION_MOTION_TIMEOUT_S = 6.0
DIRECTION_TOLERANCE_STEPS = 15
DIRECTION_STABLE_READS = 2

# Visible table nudge per amount=1.0 — fraction of each motor's calibrated span.
DEFAULT_NUDGE_SPAN_FRACTION = 0.10

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
    "elbow_up": "elbow_up",
    "elbow_down": "elbow_down",
    "hand_up": "hand_up",
    "wrist_up": "hand_up",
    "hand_down": "hand_down",
    "wrist_down": "hand_down",
    "wrist_ccw": "wrist_ccw",
    "wrist_cw": "wrist_cw",
    "hand_rotate_ccw": "wrist_ccw",
    "hand_rotate_cw": "wrist_cw",
}

# ElRobot motor step endpoints toward each table direction (not absolute targets per nudge).
# Directions are defined relative to the saved home pose; each nudge moves a fraction of span
# toward these endpoints without reaching the limit.
ELROBOT_DIRECTION_ENDPOINTS: dict[str, dict[int, int]] = {
    "up": {2: 1373, 3: 910},
    "down": {2: 2732, 3: 3186},
    "left": {1: 1176},
    "right": {1: 2920},
    "elbow_up": {4: 961},
    "elbow_down": {4: 3135},
    "hand_up": {6: 1032},
    "hand_down": {6: 3064},
    "wrist_ccw": {7: 2564},
    "wrist_cw": {7: 3932},
}

ELROBOT_MOTOR_SEMANTICS: dict[int, str] = {
    1: "base yaw — left (1176) / right (2920) on table",
    2: "shoulder — up (1373) / down (2732)",
    3: "upper arm — up (910) / down (3186)",
    4: "elbow — up (961) / down (3135)",
    6: "wrist pitch — up (1032) / down (3064)",
    7: "wrist rotate — CCW (2564) / CW (3932)",
    8: "gripper",
}


def nudge_span_fraction() -> float:
    raw = os.environ.get("NORMA_DIRECTION_NUDGE_FRACTION")
    if raw is None:
        return DEFAULT_NUDGE_SPAN_FRACTION
    value = float(raw)
    if value <= 0.0 or value > 1.0:
        raise ValueError("NORMA_DIRECTION_NUDGE_FRACTION must be between 0 and 1")
    return value


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


def _parse_motor_endpoints(raw: dict[str, Any]) -> dict[int, int]:
    motors = raw.get("motors") or raw.get("motor_endpoints") or {}
    return {int(motor_id): int(steps) for motor_id, steps in motors.items()}


def direction_endpoints_for_arm(arm_type: str) -> dict[str, dict[int, int]]:
    payload = load_direction_nudge()
    if payload is not None and payload.get("arm_type") == arm_type:
        directions = payload.get("directions") or {}
        parsed: dict[str, dict[int, int]] = {}
        for name, entry in directions.items():
            if not isinstance(entry, dict):
                continue
            endpoints = _parse_motor_endpoints(entry)
            if endpoints:
                parsed[name] = endpoints
        if parsed:
            return parsed

    if arm_type == "elrobot":
        return {name: dict(motors) for name, motors in ELROBOT_DIRECTION_ENDPOINTS.items()}

    raise RuntimeError(
        f"No direction endpoint map for arm type '{arm_type}'. "
        f"Add {DIRECTION_NUDGE_PATH} or use move_joint / move_arm_pose."
    )


def direction_calibration_payload(arm_type: str) -> dict[str, Any]:
    endpoints = direction_endpoints_for_arm(arm_type)
    fraction = nudge_span_fraction()
    payload = load_direction_nudge()
    if payload is not None and payload.get("nudge_span_fraction") is not None:
        fraction = float(payload["nudge_span_fraction"])

    home = load_home_pose()
    result: dict[str, Any] = {
        "arm_type": arm_type,
        "nudge_span_fraction": fraction,
        "directions": {
            name: {"motors": {str(m): steps for m, steps in motors.items()}}
            for name, motors in endpoints.items()
        },
        "note": (
            "Each move_direction nudge shifts primary motors toward these step endpoints "
            f"by nudge_span_fraction ({fraction:.0%}) of each motor span per amount=1.0. "
            "Semantics are relative to the saved home pose."
        ),
    }
    if arm_type == "elrobot":
        result["motor_semantics"] = {
            str(motor_id): label for motor_id, label in ELROBOT_MOTOR_SEMANTICS.items()
        }
    if home is not None:
        result["home_motor_steps"] = home.get("motor_steps")
    return result


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
    nudge_fraction: float | None = None,
) -> dict[int, int]:
    """Move primary motors a visible fraction toward direction endpoints."""
    normalized = normalize_direction(direction)
    endpoints = direction_endpoints_for_arm(arm_type)[normalized]
    fraction = nudge_fraction if nudge_fraction is not None else nudge_span_fraction()
    payload = load_direction_nudge()
    if payload is not None and payload.get("nudge_span_fraction") is not None:
        fraction = float(payload["nudge_span_fraction"])

    targets = dict(current_steps)
    for motor_id, endpoint_step in endpoints.items():
        if motor_id not in current_steps:
            continue
        current = current_steps[motor_id]
        range_min, range_max = ranges.get(motor_id, (0, 0))
        span = max(range_max - range_min, 1)
        nudge = max(1, int(round(fraction * amount * span)))

        if endpoint_step > current:
            targets[motor_id] = min(current + nudge, endpoint_step, range_max)
        elif endpoint_step < current:
            targets[motor_id] = max(current - nudge, endpoint_step, range_min)
        else:
            targets[motor_id] = current
    return targets



async def move_direction(
    session: Any,
    direction: str,
    *,
    amount: float = 1.0,
    bus_serial: str = "auto",
    wait: bool = False,
) -> dict[str, Any]:
    """Nudge the arm toward up/down/left/right using motor-range endpoints from home."""
    if amount <= 0:
        raise ValueError("amount must be positive")

    normalized = normalize_direction(direction)
    await _prepare_session(session, bus_serial)
    arm_state = session.get_arm_state(bus_serial)
    arm_type = str(arm_state.get("arm_type") or "unknown")
    current_steps = _current_joint_steps(arm_state)
    ranges = _joint_ranges(arm_state)
    endpoints = direction_endpoints_for_arm(arm_type)[normalized]
    targets = joint_step_targets_for_direction(
        current_steps,
        ranges,
        normalized,
        arm_type=arm_type,
        amount=amount,
    )

    applied_deltas = {
        motor_id: targets[motor_id] - current_steps[motor_id]
        for motor_id in endpoints
        if motor_id in targets and targets[motor_id] != current_steps[motor_id]
    }
    primary = sorted(
        applied_deltas,
        key=lambda motor_id: abs(applied_deltas[motor_id]),
        reverse=True,
    )

    await session.move_motors_steps(targets, bus_serial)

    home = load_home_pose()
    calibration = direction_calibration_payload(arm_type)
    result: dict[str, Any] = {
        "action": "move_direction",
        "direction": normalized,
        "amount": amount,
        "arm_type": arm_type,
        "nudge_span_fraction": calibration["nudge_span_fraction"],
        "direction_endpoints": {str(m): steps for m, steps in endpoints.items()},
        "motor_steps_before": {str(m): current_steps[m] for m in endpoints if m in current_steps},
        "motor_steps": {str(m): targets[m] for m in targets},
        "applied_step_deltas": applied_deltas,
        "primary_motors": primary,
        "note": (
            "Moves primary motors toward calibrated range endpoints for this direction "
            f"({calibration['nudge_span_fraction']:.0%} of span per amount=1.0). "
            "Left/right uses motor 1; up/down uses motors 2 and 3. "
            "Call get_home_pose / get_direction_calibration for reference."
        ),
    }
    if home is not None:
        result["home_motor_steps"] = home.get("motor_steps")
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
