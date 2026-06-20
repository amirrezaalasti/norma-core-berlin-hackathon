"""Inverse/forward kinematics for the ElRobot 7-DOF arm.

The IK chain is built directly from `hardware/elrobot/simulation/elrobot_follower.urdf`
and covers motors 1-7 (the arm joints) only. Its tool frame is the `Gripper_Base_v1_1`
link -- i.e. *before* the gripper's own actuation joint (motor 8). Gripper open/close is
not part of this module; it's handled entirely by StationSession.set_gripper /
open_gripper / close_gripper.

`base_elements` truncation note: ikpy's `Chain.from_urdf_file` does not stop chain
traversal at a given link name -- once explicit `base_elements` are exhausted it keeps
auto-walking the URDF tree. The reliable way to get a 7-joint arm-only chain is to build
the full chain and then truncate the resulting `Chain.links` list by slicing, which is
what `_build_elrobot_chain` does. Do not "simplify" this back to a `base_elements`-only
call -- that was tried and does not stop where you'd expect.

Coordination flag for whoever produces target_xyz (vision/perception): `target_xyz` must
already be expressed in the robot's `base_link` frame, in meters, to match this URDF. If
that's a camera-frame coordinate, a camera-to-base_link transform must be composed before
calling into this module -- hand-eye calibration is out of scope here and not implemented.

Hardware-unverifiable assumption: `radians_to_normalized` / `normalized_to_radians` assume
normalized 0.0 corresponds to a joint's URDF lower bound and normalized 1.0 to its URDF
upper bound, linearly. The URDF limits are nominal CAD values; the real per-motor range
comes from runtime auto-calibration against physical hard stops (exposed as range_min/
range_max in motor_state.py, sourced from the servo's own MIN_ANGLE_LIMIT/MAX_ANGLE_LIMIT
EEPROM registers). This cannot be checked without the physical arm. Verify once hardware
is available by commanding two well-separated targets via move_to_xyz and confirming the
gripper lands in the right place, not just that it moves.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
from ikpy.chain import Chain

from .paths import REPO_ROOT

ELROBOT_URDF_PATH = REPO_ROOT / "hardware" / "elrobot" / "simulation" / "elrobot_follower.urdf"

# Index into the truncated 15-link ikpy chain for each arm joint's revolute link,
# keyed by motor id. Verified against the URDF: rev_motor_01..rev_motor_07 == motor 1-7.
_ACTIVE_LINK_INDEX_BY_MOTOR: dict[int, int] = {1: 2, 2: 4, 3: 6, 4: 8, 5: 10, 6: 12, 7: 14}
_TIP_CHAIN_LENGTH = 15

IK_POSITION_TOLERANCE_M = 0.005


def _build_elrobot_chain() -> Chain:
    # from_urdf_file defaults every link (including fixed joints) to active and warns
    # about each one before we discard that default mask in favor of our own below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        full_chain = Chain.from_urdf_file(str(ELROBOT_URDF_PATH))
    tip_links = full_chain.links[:_TIP_CHAIN_LENGTH]
    mask = [False] * _TIP_CHAIN_LENGTH
    for index in _ACTIVE_LINK_INDEX_BY_MOTOR.values():
        mask[index] = True
    return Chain(tip_links, active_links_mask=mask, name="elrobot_tip")


@functools.lru_cache(maxsize=1)
def get_elrobot_chain() -> Chain:
    """Cached 7-DOF position-IK chain for ElRobot. Built lazily on first use."""
    return _build_elrobot_chain()


def _active_link_index(motor_id: int) -> int:
    try:
        return _ACTIVE_LINK_INDEX_BY_MOTOR[motor_id]
    except KeyError:
        raise ValueError(f"motor_id {motor_id} is not an ElRobot arm joint (valid: 1-7)") from None


def joint_bounds_rad(motor_id: int) -> tuple[float, float]:
    """URDF-derived (lower, upper) radian limit for one arm joint (motor id 1-7)."""
    index = _active_link_index(motor_id)
    return get_elrobot_chain().links[index].bounds


def _full_vector_from_joint_rad(joint_rad: dict[int, float]) -> list[float]:
    unknown = set(joint_rad) - set(_ACTIVE_LINK_INDEX_BY_MOTOR)
    if unknown:
        raise ValueError(f"unknown motor ids {sorted(unknown)}; valid arm joints are 1-7")
    vector = [0.0] * _TIP_CHAIN_LENGTH
    for motor_id, angle in joint_rad.items():
        vector[_active_link_index(motor_id)] = angle
    return vector


def forward_kinematics(joint_rad: dict[int, float]) -> tuple[float, float, float]:
    """Gripper-base XYZ (meters, base_link frame) for the given joint angles.

    joint_rad is keyed by motor id 1-7; missing keys default to 0.0 radians.
    """
    vector = _full_vector_from_joint_rad(joint_rad)
    transform = get_elrobot_chain().forward_kinematics(vector)
    return tuple(float(v) for v in transform[:3, 3])


def _validate_target_xyz(target_xyz: Any) -> np.ndarray:
    try:
        values = [float(v) for v in target_xyz]
    except (TypeError, ValueError):
        raise ValueError(f"target_xyz must be 3 finite numbers, got {target_xyz!r}") from None
    if len(values) != 3 or not all(np.isfinite(v) for v in values):
        raise ValueError(f"target_xyz must be 3 finite numbers, got {target_xyz!r}")
    return np.array(values, dtype=float)


def solve_ik(
    target_xyz: Any,
    *,
    initial_joint_rad: dict[int, float] | None = None,
    tolerance_m: float = IK_POSITION_TOLERANCE_M,
) -> dict[int, float]:
    """Solve position-only IK for the ElRobot 7-DOF arm.

    Parameters
    ----------
    target_xyz: [x, y, z] in meters, in the URDF's base_link frame.
    initial_joint_rad: optional optimizer seed, keyed by motor id 1-7 (radians).
        Defaults to the all-zero home pose if omitted.
    tolerance_m: max acceptable distance between the solved pose's FK position and
        target_xyz before this raises.

    Returns
    -------
    dict[int, float]: motor_id (1-7) -> joint angle in radians.

    Raises
    ------
    ValueError: target_xyz is not a 3-element finite numeric sequence.
    RuntimeError: IK did not converge within tolerance_m (target likely outside the
        arm's reachable workspace), or a solved joint angle falls outside that joint's
        URDF bounds.
    """
    target = _validate_target_xyz(target_xyz)
    chain = get_elrobot_chain()
    initial_vector = _full_vector_from_joint_rad(initial_joint_rad or {})

    # ikpy's inverse_kinematics never raises for unreachable targets -- it just returns
    # its best-effort result with a large residual, so the FK-of-solution check below is
    # mandatory, not a nice-to-have.
    solution = chain.inverse_kinematics(target, initial_position=initial_vector)
    fk = np.array(chain.forward_kinematics(solution)[:3, 3])
    residual = float(np.linalg.norm(fk - target))
    if residual > tolerance_m:
        raise RuntimeError(
            f"IK did not converge for target {tuple(target.tolist())}: residual "
            f"{residual:.4f} m exceeds tolerance {tolerance_m} m (target may be outside "
            "the arm's reachable workspace)"
        )

    joint_rad: dict[int, float] = {}
    for motor_id, index in _ACTIVE_LINK_INDEX_BY_MOTOR.items():
        angle = float(solution[index])
        lower, upper = chain.links[index].bounds
        if not (lower - 1e-6 <= angle <= upper + 1e-6):
            raise RuntimeError(
                f"IK solution for motor {motor_id} is out of bounds: {angle:.4f} rad "
                f"not in [{lower:.4f}, {upper:.4f}]"
            )
        joint_rad[motor_id] = angle
    return joint_rad


def radians_to_normalized(angle_rad: float, motor_id: int) -> float:
    """Convert an IK-solved joint angle (radians) to the 0.0-1.0 normalized convention
    used by StationSession.move_motors_normalized.

    Assumes linear interpolation against this joint's URDF bounds: normalized 0.0 ==
    URDF lower limit, 1.0 == URDF upper limit. See module docstring -- this assumption
    needs hardware verification and cannot be checked right now.

    Raises ValueError if angle_rad falls outside this joint's URDF bounds by more than a
    tiny floating-point epsilon (no silent clamping of a real out-of-range angle).
    """
    lower, upper = joint_bounds_rad(motor_id)
    span = upper - lower
    epsilon = 1e-6
    if angle_rad < lower - epsilon or angle_rad > upper + epsilon:
        raise ValueError(
            f"angle {angle_rad:.4f} rad for motor {motor_id} is outside URDF bounds "
            f"[{lower:.4f}, {upper:.4f}]"
        )
    return min(1.0, max(0.0, (angle_rad - lower) / span))


def normalized_to_radians(position: float, motor_id: int) -> float:
    """Inverse of radians_to_normalized -- converts a motor's current normalized position
    (e.g. from get_arm_state) into a joint angle in radians. Used by get_gripper_xyz.
    """
    if position < 0.0 or position > 1.0:
        raise ValueError(f"position must be between 0.0 and 1.0, got {position}")
    lower, upper = joint_bounds_rad(motor_id)
    return lower + (upper - lower) * position


def joint_rad_dict_to_normalized(joint_rad: dict[int, float]) -> dict[int, float]:
    """Vectorized convenience: apply radians_to_normalized to every entry, keyed by motor id."""
    return {motor_id: radians_to_normalized(angle, motor_id) for motor_id, angle in joint_rad.items()}
