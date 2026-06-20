from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArmProfile:
    """Maps physical arm layout to motor ids on the ST3215 bus."""

    name: str
    motor_count: int
    joint_motor_ids: tuple[int, ...]
    gripper_motor_id: int | None

    @property
    def label(self) -> str:
        labels = {
            "so101": "SO-101 (6 DoF + gripper on motor 6)",
            "elrobot": "ElRobot (7 DoF + gripper on motor 8)",
        }
        return labels.get(self.name, f"Generic arm ({self.motor_count} motors)")


def detect_arm_profile(motor_ids: list[int]) -> ArmProfile:
    """Infer arm type from connected motor ids."""
    sorted_ids = tuple(sorted(motor_ids))
    count = len(sorted_ids)

    if count == 6 and sorted_ids == (1, 2, 3, 4, 5, 6):
        return ArmProfile("so101", 6, (1, 2, 3, 4, 5), 6)
    if count == 8 and sorted_ids == (1, 2, 3, 4, 5, 6, 7, 8):
        return ArmProfile("elrobot", 8, (1, 2, 3, 4, 5, 6, 7), 8)
    if count >= 2:
        return ArmProfile(
            "unknown",
            count,
            sorted_ids[:-1],
            sorted_ids[-1],
        )
    if count == 1:
        return ArmProfile("unknown", count, (), sorted_ids[0])
    return ArmProfile("unknown", 0, (), None)


def motor_role(profile: ArmProfile, motor_id: int) -> str:
    if profile.gripper_motor_id is not None and motor_id == profile.gripper_motor_id:
        return "gripper"
    if motor_id in profile.joint_motor_ids:
        return f"joint_{motor_id}"
    return "motor"


def annotate_motor(motor: dict[str, Any], profile: ArmProfile) -> dict[str, Any]:
    motor_id = motor["motor_id"]
    annotated = {
        **motor,
        "role": motor_role(profile, motor_id),
    }
    range_min = motor["range_min"]
    range_max = motor["range_max"]
    if range_min > 0 or range_max > 0:
        if range_min < range_max:
            span = range_max - range_min
            present = motor["present_position"]
            annotated["present_position_normalized"] = round(
                (present - range_min) / span, 4
            )
            annotated["target_position_normalized"] = round(
                (motor["target_position"] - range_min) / span, 4
            )
    return annotated
