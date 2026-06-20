from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from typing import Any

RAM_TORQUE_ENABLE = 0x28
RAM_ACC = 0x29
RAM_GOAL_POSITION = 0x2A
RAM_GOAL_SPEED = 0x2E
RAM_PRESENT_POSITION = 0x38
RAM_STATUS = 0x40
RAM_PRESENT_CURRENT = 0x45

MAX_ANGLE_STEP = 4095
SIGN_BIT_MASK = 0x8000


def _u16(state: bytes, addr: int) -> int:
    if len(state) < addr + 2:
        return 0
    return struct.unpack_from("<H", state, addr)[0]


def _u8(state: bytes, addr: int) -> int:
    if len(state) <= addr:
        return 0
    return state[addr]


def _normal_position(raw: int) -> int:
    if raw & SIGN_BIT_MASK:
        magnitude = raw & MAX_ANGLE_STEP
        return (MAX_ANGLE_STEP + 1 - magnitude) & MAX_ANGLE_STEP
    return raw & MAX_ANGLE_STEP


@dataclass
class MotorSnapshot:
    motor_id: int
    present_position: int
    target_position: int
    range_min: int
    range_max: int
    torque_enabled: bool
    present_current: int
    goal_speed: int
    goal_accel: int
    error_status: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_motor_snapshot(motor_reader) -> MotorSnapshot:
    state_bytes = bytes(motor_reader.get_state())
    present = _normal_position(_u16(state_bytes, RAM_PRESENT_POSITION))
    goal = _normal_position(_u16(state_bytes, RAM_GOAL_POSITION))
    torque_on = _u8(state_bytes, RAM_TORQUE_ENABLE) != 0

    return MotorSnapshot(
        motor_id=motor_reader.get_id(),
        present_position=present,
        target_position=goal if torque_on else present,
        range_min=motor_reader.get_range_min(),
        range_max=motor_reader.get_range_max(),
        torque_enabled=torque_on,
        present_current=_u16(state_bytes, RAM_PRESENT_CURRENT),
        goal_speed=_u16(state_bytes, RAM_GOAL_SPEED),
        goal_accel=_u8(state_bytes, RAM_ACC),
        error_status=_u8(state_bytes, RAM_STATUS),
    )


def resolve_bus_serial(inference_state, requested: str) -> str:
    buses = inference_state.get_buses() or []
    if not buses:
        raise RuntimeError("No ST3215 buses reported by station")

    if requested == "auto":
        if len(buses) != 1:
            serials = [
                b.get_bus().get_serial_number()
                for b in buses
                if b.get_bus()
            ]
            raise RuntimeError(
                f"bus_serial='auto' requires exactly one bus, found {len(buses)}: {serials}"
            )
        info = buses[0].get_bus()
        if info is None:
            raise RuntimeError("Bus has no metadata")
        return info.get_serial_number()

    for bus in buses:
        info = bus.get_bus()
        if info and info.get_serial_number() == requested:
            return requested
    raise RuntimeError(f"Bus '{requested}' not found on station")


def find_bus(inference_state, bus_serial: str):
    if inference_state is None:
        return None
    for bus in inference_state.get_buses() or []:
        info = bus.get_bus()
        if info and info.get_serial_number() == bus_serial:
            return bus
    return None


def normalized_to_steps(position: float, range_min: int, range_max: int) -> int:
    if position < 0.0 or position > 1.0:
        raise ValueError("position must be between 0.0 and 1.0")
    if range_min == 0 and range_max == 0:
        raise ValueError("motor is not calibrated (range_min and range_max are both 0)")
    if range_min >= range_max:
        raise ValueError(f"invalid calibrated range [{range_min}, {range_max}]")
    return int(range_min + (range_max - range_min) * position)
