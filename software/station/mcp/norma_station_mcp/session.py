from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from .arm_model import ArmProfile, annotate_motor, detect_arm_profile, motor_role
from .motor_state import (
    find_bus,
    normalized_to_steps,
    parse_motor_snapshot,
    resolve_bus_serial,
)
from .paths import setup_import_paths

setup_import_paths()

try:
    from station_py import new_station_client, send_commands
    from target.gen_python.protobuf.drivers.st3215 import st3215
    from target.gen_python.protobuf.station import commands, drivers
except ImportError as exc:
    raise ImportError(
        "Missing generated protobufs or station_py. "
        "From the repo root run: make protobuf"
    ) from exc


def _station_host() -> str:
    return os.environ.get("STATION_HOST", "localhost:8888")


class StationSession:
    """Persistent TCP client with a cached st3215/inference frame."""

    def __init__(self, host: str | None = None):
        self.host = host or _station_host()
        self.logger = logging.getLogger("norma-station-mcp")
        self.client = None
        self.latest_inference: st3215.InferenceStateReader | None = None
        self.latest_stamp_s: float | None = None
        self.frame_count = 0
        self._init_lock = asyncio.Lock()
        self._follow_task: asyncio.Task | None = None
        self._queue: asyncio.Queue | None = None
        self._last_error: str | None = None

    async def ensure_connected(self) -> None:
        async with self._init_lock:
            if self.client is not None:
                return

            self.logger.info("Connecting to station at %s", self.host)
            self.client = await new_station_client(self.host, self.logger)
            self._queue = asyncio.Queue()
            self._follow_task = asyncio.create_task(self._follow_inference())
            self.logger.info("Connected to station")

    async def _follow_inference(self) -> None:
        assert self.client is not None
        assert self._queue is not None

        error_queue = self.client.follow("st3215/inference", self._queue)

        while True:
            if not error_queue.empty():
                err = error_queue.get_nowait()
                self._last_error = str(err)
                self.logger.error("Inference stream error: %s", err)
                return

            entry = await self._queue.get()
            if entry is None:
                self._last_error = "Inference stream closed"
                return

            try:
                self.latest_inference = st3215.InferenceStateReader(entry.Data)
                self.latest_stamp_s = time.monotonic()
                self.frame_count += 1
            except Exception as exc:
                self.logger.exception("Failed to decode inference frame: %s", exc)

    async def wait_for_inference(self, timeout_s: float = 10.0) -> None:
        await self.ensure_connected()
        deadline = time.monotonic() + timeout_s
        while self.latest_inference is None:
            if self._last_error:
                raise RuntimeError(self._last_error)
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"No st3215/inference frames received within {timeout_s}s. "
                    "Is station running with --tcp and the ST3215 driver enabled?"
                )
            await asyncio.sleep(0.05)

    async def _ensure_ready(self) -> None:
        """Connect and wait for at least one inference frame before bus/motor lookups."""
        await self.ensure_connected()
        await self.wait_for_inference()

    def connection_info(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "connected": self.client is not None and self.client.connected,
            "setup_done": self.client.setup_done if self.client else False,
            "frame_count": self.frame_count,
            "has_latest_inference": self.latest_inference is not None,
            "last_error": self._last_error,
        }

    def _resolve_bus(self, bus_serial: str = "auto") -> tuple[str, Any, ArmProfile]:
        if self.latest_inference is None:
            raise RuntimeError("No inference data available yet")

        resolved_serial = resolve_bus_serial(self.latest_inference, bus_serial)
        bus = find_bus(self.latest_inference, resolved_serial)
        if bus is None:
            raise RuntimeError(f"Bus '{resolved_serial}' missing from latest frame")

        motor_ids = [m.get_id() for m in (bus.get_motors() or [])]
        profile = detect_arm_profile(motor_ids)
        return resolved_serial, bus, profile

    def get_arm_state(self, bus_serial: str = "auto") -> dict[str, Any]:
        """Return arm-oriented view: joints, gripper, and detected arm type."""
        resolved_serial, bus, profile = self._resolve_bus(bus_serial)

        joints = []
        gripper = None
        other = []

        for motor in bus.get_motors() or []:
            snapshot = annotate_motor(parse_motor_snapshot(motor).to_dict(), profile)
            role = snapshot["role"]
            if role == "gripper":
                gripper = snapshot
            elif role.startswith("joint_"):
                joints.append(snapshot)
            else:
                other.append(snapshot)

        joints.sort(key=lambda item: item["motor_id"])

        return {
            "bus_serial": resolved_serial,
            "arm_type": profile.name,
            "arm_label": profile.label,
            "joint_motor_ids": list(profile.joint_motor_ids),
            "gripper_motor_id": profile.gripper_motor_id,
            "joints": joints,
            "gripper": gripper,
            "other_motors": other,
            "note": (
                "Positions are joint-space (per-motor normalized 0.0-1.0), "
                "not Cartesian XYZ. Use move_joint / move_arm_pose for motion."
            ),
        }

    def list_buses(self) -> list[dict[str, Any]]:
        if self.latest_inference is None:
            return []

        buses = []
        for bus_state in self.latest_inference.get_buses() or []:
            info = bus_state.get_bus()
            if info is None:
                continue
            motors = [
                parse_motor_snapshot(motor).to_dict()
                for motor in (bus_state.get_motors() or [])
            ]
            buses.append(
                {
                    "serial": info.get_serial_number(),
                    "motor_count": len(motors),
                    "motors": motors,
                }
            )
        return buses

    def get_motor(
        self, bus_serial: str = "auto", motor_id: int = 1
    ) -> dict[str, Any]:
        if self.latest_inference is None:
            raise RuntimeError("No inference data available yet")

        resolved_serial = resolve_bus_serial(self.latest_inference, bus_serial)
        bus = find_bus(self.latest_inference, resolved_serial)
        if bus is None:
            raise RuntimeError(f"Bus '{resolved_serial}' missing from latest frame")

        for motor in bus.get_motors() or []:
            if motor.get_id() == motor_id:
                snapshot = parse_motor_snapshot(motor)
                return {
                    "bus_serial": resolved_serial,
                    **snapshot.to_dict(),
                }

        available = [m.get_id() for m in (bus.get_motors() or [])]
        raise RuntimeError(
            f"Motor {motor_id} not found on bus '{resolved_serial}'. "
            f"Available motor ids: {available}"
        )

    def _sync_write_command(
        self, bus_serial: str, address: int, motor_writes: list[tuple[int, bytes]]
    ) -> commands.DriverCommand:
        return commands.DriverCommand(
            type=drivers.StationCommandType.STC_ST3215_COMMAND,
            body=st3215.Command(
                target_bus_serial=bus_serial,
                sync_write=st3215.ST3215SyncWriteCommand(
                    address=address,
                    motors=[
                        st3215.ST3215SyncWriteCommand_MotorWrite(
                            motor_id=motor_id,
                            value=value,
                        )
                        for motor_id, value in motor_writes
                    ],
                ),
            ).encode(),
        )

    async def move_motors_normalized(
        self,
        positions: dict[int, float],
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Move multiple motors in one sync_write batch."""
        positions = {int(motor_id): float(position) for motor_id, position in positions.items()}
        await self.ensure_connected()
        await self.wait_for_inference()

        if not positions:
            raise ValueError("positions must not be empty")

        resolved_serial, _, profile = self._resolve_bus(bus_serial)
        goal_writes: list[tuple[int, bytes]] = []
        resolved_targets: dict[str, dict[str, Any]] = {}

        for motor_id, position in sorted(positions.items()):
            if position < 0.0 or position > 1.0:
                raise ValueError(
                    f"position for motor {motor_id} must be between 0.0 and 1.0"
                )
            motor = self.get_motor(resolved_serial, motor_id)
            steps = normalized_to_steps(
                position, motor["range_min"], motor["range_max"]
            )
            goal_writes.append((motor_id, steps.to_bytes(2, byteorder="little")))
            resolved_targets[str(motor_id)] = {
                "role": motor_role(profile, motor_id),
                "position_normalized": position,
                "sent_steps": steps,
            }

        await send_commands(
            self.client,
            [self._sync_write_command(resolved_serial, 0x2A, goal_writes)],
        )

        return {
            "bus_serial": resolved_serial,
            "arm_type": profile.name,
            "motors": resolved_targets,
        }

    async def move_joint(
        self,
        joint_id: int,
        position: float,
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Move one arm joint (joint id equals motor id on SO-101 / ElRobot)."""
        await self._ensure_ready()
        _, _, profile = self._resolve_bus(bus_serial)
        if joint_id not in profile.joint_motor_ids:
            raise ValueError(
                f"Joint {joint_id} is not an arm joint for {profile.label}. "
                f"Valid joints: {list(profile.joint_motor_ids)}. "
                f"Use set_gripper / open_gripper / close_gripper for the gripper."
            )
        result = await self.move_motors_normalized({joint_id: position}, bus_serial)
        result["joint_id"] = joint_id
        return result

    async def move_arm_pose(
        self,
        joint_positions: dict[int, float],
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Move multiple arm joints at once. Keys are joint ids (motor ids 1-5 or 1-7)."""
        joint_positions = {int(joint_id): float(position) for joint_id, position in joint_positions.items()}
        await self._ensure_ready()
        _, _, profile = self._resolve_bus(bus_serial)
        invalid = [
            joint_id
            for joint_id in joint_positions
            if joint_id not in profile.joint_motor_ids
        ]
        if invalid:
            raise ValueError(
                f"Invalid joint ids {invalid} for {profile.label}. "
                f"Valid joints: {list(profile.joint_motor_ids)}. "
                "Gripper is controlled separately."
            )
        result = await self.move_motors_normalized(joint_positions, bus_serial)
        result["joint_positions"] = joint_positions
        return result

    async def set_gripper(
        self,
        position: float,
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Set gripper opening. 0.0 = fully closed, 1.0 = fully open (calibrated range)."""
        await self._ensure_ready()
        _, _, profile = self._resolve_bus(bus_serial)
        if profile.gripper_motor_id is None:
            raise RuntimeError("No gripper motor detected on this bus")

        result = await self.move_motors_normalized(
            {profile.gripper_motor_id: position},
            bus_serial,
        )
        result["gripper_motor_id"] = profile.gripper_motor_id
        result["gripper_position"] = position
        result["gripper_state"] = "open" if position >= 0.9 else "closed" if position <= 0.1 else "partial"
        return result

    async def open_gripper(self, bus_serial: str = "auto") -> dict[str, Any]:
        return await self.set_gripper(1.0, bus_serial)

    async def close_gripper(self, bus_serial: str = "auto") -> dict[str, Any]:
        return await self.set_gripper(0.0, bus_serial)

    async def wait_for_arm_pose(
        self,
        joint_targets: dict[int, float],
        *,
        tolerance: float = 0.015,
        stable_reads: int = 5,
        poll_s: float = 0.15,
        timeout_s: float = 45.0,
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Block until all listed joints reach their targets (within tolerance)."""
        targets = {int(joint_id): float(position) for joint_id, position in joint_targets.items()}
        deadline = time.monotonic() + timeout_s
        consecutive_ok = 0
        last_state: dict[str, Any] | None = None

        while time.monotonic() < deadline:
            await self.wait_for_inference(timeout_s=min(2.0, deadline - time.monotonic()))
            last_state = self.get_arm_state(bus_serial)
            joint_by_id = {
                joint["motor_id"]: joint for joint in last_state.get("joints", [])
            }
            max_error = 0.0
            for joint_id, target in targets.items():
                joint = joint_by_id.get(joint_id)
                if joint is None:
                    raise RuntimeError(f"Joint {joint_id} missing from arm state")
                present = float(joint["present_position_normalized"])
                max_error = max(max_error, abs(present - target))

            if max_error <= tolerance:
                consecutive_ok += 1
                if consecutive_ok >= stable_reads:
                    return {
                        "reached": True,
                        "max_error": max_error,
                        "joint_targets": targets,
                        "arm_state": last_state,
                    }
            else:
                consecutive_ok = 0

            await asyncio.sleep(poll_s)

        return {
            "reached": False,
            "max_error": max_error,
            "joint_targets": targets,
            "arm_state": last_state,
            "note": f"Timed out after {timeout_s}s waiting for arm pose",
        }

    async def enable_arm_torque(self, bus_serial: str = "auto") -> dict[str, Any]:
        await self._ensure_ready()
        _, bus, _profile = self._resolve_bus(bus_serial)
        motor_ids = [m.get_id() for m in (bus.get_motors() or [])]
        return await self.set_torque(motor_ids, True, bus_serial)

    async def disable_arm_torque(self, bus_serial: str = "auto") -> dict[str, Any]:
        await self._ensure_ready()
        _, bus, _profile = self._resolve_bus(bus_serial)
        motor_ids = [m.get_id() for m in (bus.get_motors() or [])]
        return await self.set_torque(motor_ids, False, bus_serial)

    async def move_motor_steps(
        self,
        motor_id: int,
        goal_steps: int,
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        await self.ensure_connected()
        await self.wait_for_inference()

        resolved_serial = resolve_bus_serial(self.latest_inference, bus_serial)
        motor = self.get_motor(resolved_serial, motor_id)
        range_min = motor["range_min"]
        range_max = motor["range_max"]

        if range_min > 0 or range_max > 0:
            clamped = max(range_min, min(range_max, goal_steps))
        else:
            clamped = goal_steps

        cmd = commands.DriverCommand(
            type=drivers.StationCommandType.STC_ST3215_COMMAND,
            body=st3215.Command(
                target_bus_serial=resolved_serial,
                write=st3215.ST3215WriteCommand(
                    motor_id=motor_id,
                    address=0x2A,
                    value=clamped.to_bytes(2, byteorder="little"),
                ),
            ).encode(),
        )
        await send_commands(self.client, [cmd])

        return {
            "bus_serial": resolved_serial,
            "motor_id": motor_id,
            "requested_steps": goal_steps,
            "sent_steps": clamped,
        }

    async def move_motor_normalized(
        self,
        motor_id: int,
        position: float,
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        await self.wait_for_inference()
        resolved_serial = resolve_bus_serial(self.latest_inference, bus_serial)
        motor = self.get_motor(resolved_serial, motor_id)
        goal_steps = normalized_to_steps(
            position, motor["range_min"], motor["range_max"]
        )
        result = await self.move_motor_steps(motor_id, goal_steps, resolved_serial)
        result["position_normalized"] = position
        return result

    async def set_torque(
        self,
        motor_ids: list[int],
        enable: bool,
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        await self.ensure_connected()
        await self.wait_for_inference()

        resolved_serial = resolve_bus_serial(self.latest_inference, bus_serial)
        if not motor_ids:
            raise ValueError("motor_ids must not be empty")

        value = b"\x01" if enable else b"\x00"
        cmd = commands.DriverCommand(
            type=drivers.StationCommandType.STC_ST3215_COMMAND,
            body=st3215.Command(
                target_bus_serial=resolved_serial,
                sync_write=st3215.ST3215SyncWriteCommand(
                    address=0x28,
                    motors=[
                        st3215.ST3215SyncWriteCommand_MotorWrite(
                            motor_id=motor_id,
                            value=value,
                        )
                        for motor_id in motor_ids
                    ],
                ),
            ).encode(),
        )
        await send_commands(self.client, [cmd])

        return {
            "bus_serial": resolved_serial,
            "motor_ids": motor_ids,
            "torque_enabled": enable,
        }


_session: StationSession | None = None


def get_session() -> StationSession:
    global _session
    if _session is None:
        _session = StationSession()
    return _session
