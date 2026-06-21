from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from typing import Any

from . import kinematics
from .arm_model import ArmProfile, annotate_motor, detect_arm_profile, motor_role
from .motor_state import (
    find_bus,
    normalized_to_steps,
    parse_motor_snapshot,
    resolve_bus_serial,
)
from .paths import setup_import_paths

setup_import_paths()

STATION_MOCK = os.environ.get("STATION_MOCK") == "1"

try:
    if STATION_MOCK:
        from .fake_station import new_station_client, send_commands
    else:
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


# Temporary testing safety cap -- remove or raise once movement direction/magnitude has
# been verified against real hardware (see kinematics.py module docstring for the
# unverified normalized<->radians calibration assumption this guards against).
TESTING_MAX_MOVE_DELTA_M = 0.05

# Placeholder workspace floor, in meters, relative to HOME_POSE_XYZ (see kinematics.py's
# "Origin/home convention" -- HOME_POSE_XYZ is the *gripper's* position at rest, not the
# robot's base). kinematics._home_offset_xyz()'s z component (~0.110m) is how far above
# the base_link origin the gripper sits at home; -0.10 approximates "don't go below
# roughly the base/mounting level" (a reasonable proxy for table height when the base is
# table-mounted) with a little margin. This is still an uncalibrated approximation, not
# a measured real table height -- recalibrate once that's actually measured on hardware.
MIN_SAFE_Z_M = -0.10


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

    def _check_workspace_floor_for_joint_move(
        self,
        positions: dict[int, float],
        profile: ArmProfile,
        resolved_serial: str,
    ) -> None:
        """Reject a joint-space move that would put the gripper below MIN_SAFE_Z_M.

        Joint-space moves (move_joint/move_arm_pose/sweep-style demos) had no floor
        protection at all -- only the Cartesian path (move_to_xyz) did -- even though a
        joint-space move can just as easily drive the arm into the table. This mirrors
        move_to_xyz's floor check, just computed via forward kinematics over the
        *resulting* full joint configuration (proposed positions merged over the arm's
        current state) instead of taking Z directly from the caller.
        """
        if profile.name != "elrobot":
            return
        arm_joint_ids = set(profile.joint_motor_ids)
        proposed_arm_positions = {k: v for k, v in positions.items() if k in arm_joint_ids}
        if not proposed_arm_positions:
            return  # gripper-only move -- the gripper isn't part of the IK chain

        arm_state = self.get_arm_state(resolved_serial)
        joint_rad: dict[int, float] = {}
        for joint in arm_state["joints"]:
            motor_id = joint["motor_id"]
            if motor_id not in proposed_arm_positions and "present_position_normalized" not in joint:
                raise RuntimeError(
                    f"Motor {motor_id} is not calibrated (no range_min/range_max); "
                    "cannot verify the workspace safety floor for this move"
                )
            normalized = proposed_arm_positions.get(motor_id, joint.get("present_position_normalized"))
            joint_rad[motor_id] = kinematics.normalized_to_radians(normalized, motor_id)

        xyz = kinematics.forward_kinematics(joint_rad)
        if xyz[2] < MIN_SAFE_Z_M:
            raise RuntimeError(
                f"Refusing move: resulting position {list(xyz)} has z={xyz[2]:.3f} m, "
                f"below the workspace safety floor of {MIN_SAFE_Z_M} m. See "
                "MIN_SAFE_Z_M in session.py -- this is an uncalibrated placeholder, not "
                "a measured table height."
            )

    async def move_motors_normalized(
        self,
        positions: dict[int, float],
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Move multiple motors in one sync_write batch."""
        await self.ensure_connected()
        await self.wait_for_inference()

        if not positions:
            raise ValueError("positions must not be empty")

        resolved_serial, _, profile = self._resolve_bus(bus_serial)
        self._check_workspace_floor_for_joint_move(positions, profile, resolved_serial)
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
        """Set gripper opening. 0.0 = fully open, 1.0 = fully closed (calibrated range)."""
        _, _, profile = self._resolve_bus(bus_serial)
        if profile.gripper_motor_id is None:
            raise RuntimeError("No gripper motor detected on this bus")

        result = await self.move_motors_normalized(
            {profile.gripper_motor_id: position},
            bus_serial,
        )
        result["gripper_motor_id"] = profile.gripper_motor_id
        result["gripper_position"] = position
        result["gripper_state"] = "closed" if position >= 0.9 else "open" if position <= 0.1 else "partial"
        return result

    async def open_gripper(self, bus_serial: str = "auto") -> dict[str, Any]:
        return await self.set_gripper(0.0, bus_serial)

    async def close_gripper(self, bus_serial: str = "auto") -> dict[str, Any]:
        return await self.set_gripper(1.0, bus_serial)

    async def enable_arm_torque(self, bus_serial: str = "auto") -> dict[str, Any]:
        _, bus, _profile = self._resolve_bus(bus_serial)
        motor_ids = [m.get_id() for m in (bus.get_motors() or [])]
        return await self.set_torque(motor_ids, True, bus_serial)

    async def disable_arm_torque(self, bus_serial: str = "auto") -> dict[str, Any]:
        _, bus, _profile = self._resolve_bus(bus_serial)
        motor_ids = [m.get_id() for m in (bus.get_motors() or [])]
        return await self.set_torque(motor_ids, False, bus_serial)

    def _require_elrobot(self, profile: ArmProfile) -> None:
        if profile.name != "elrobot":
            raise RuntimeError(
                f"Cartesian/IK control is only supported for ElRobot; detected arm is {profile.label}"
            )

    async def move_to_xyz(
        self,
        target_xyz: list[float] | tuple[float, float, float],
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Move the gripper to a Cartesian point via inverse kinematics.

        target_xyz: [x, y, z] in meters, in the robot's base_link frame (see
        kinematics.py module docstring for the frame/calibration caveats). ElRobot
        only -- raises if the detected arm has no IK chain (e.g. SO-101).

        Capped by TESTING_MAX_MOVE_DELTA_M: rejects (before solving IK or sending any
        motor command) a target more than that far from the gripper's current position.
        Temporary guardrail while the normalized<->radians calibration mapping is
        unverified against real hardware -- raise or remove once it's confirmed.

        Also rejects targets below MIN_SAFE_Z_M (a placeholder workspace floor -- see
        that constant's docstring).

        Seeds the IK solve with the arm's current joint configuration (not the all-zero
        default) -- this is a 7-DOF arm solving a 3-DOF position-only target, so there's
        a continuous family of joint configurations reaching any given point. Without a
        seed near the current pose, the optimizer converges to an arbitrary point in
        that redundant space on every call, independent of the previous one -- visibly,
        joints with no real reason to move (e.g. wrist roll) would jump around between
        consecutive small Cartesian moves. Seeding from "where the arm already is" makes
        the solver converge to the *nearest* valid solution instead, so small moves stay
        smooth and redundant joints don't wander without cause.
        """
        await self.ensure_connected()
        await self.wait_for_inference()
        resolved_serial, _, profile = self._resolve_bus(bus_serial)
        self._require_elrobot(profile)

        target = [float(v) for v in target_xyz]
        if target[2] < MIN_SAFE_Z_M:
            raise RuntimeError(
                f"Refusing move: target {target} has z={target[2]:.3f} m, below the "
                f"workspace safety floor of {MIN_SAFE_Z_M} m. See MIN_SAFE_Z_M in "
                "session.py -- this is an uncalibrated placeholder, not a measured "
                "table height."
            )
        current = await self.get_gripper_xyz(resolved_serial)
        delta_m = math.dist(current["xyz"], target)
        if delta_m > TESTING_MAX_MOVE_DELTA_M:
            raise RuntimeError(
                f"Refusing move: target {target} is {delta_m:.3f} m from the current "
                f"gripper position {current['xyz']}, exceeding the temporary testing "
                f"limit of {TESTING_MAX_MOVE_DELTA_M} m. Raise TESTING_MAX_MOVE_DELTA_M "
                "in session.py once movement direction/magnitude has been verified on "
                "real hardware."
            )

        joint_rad = kinematics.solve_ik(target, initial_joint_rad=current["joint_rad"])
        normalized = kinematics.joint_rad_dict_to_normalized(joint_rad)
        result = await self.move_motors_normalized(normalized, resolved_serial)
        result["target_xyz"] = target
        result["joint_rad"] = joint_rad
        return result

    async def get_gripper_xyz(self, bus_serial: str = "auto") -> dict[str, Any]:
        """Read the gripper's current Cartesian position (meters, base_link frame) via
        forward kinematics from the arm's current joint positions. ElRobot only.
        """
        await self.ensure_connected()
        await self.wait_for_inference()
        arm_state = self.get_arm_state(bus_serial)
        if arm_state["arm_type"] != "elrobot":
            raise RuntimeError(
                "Cartesian/IK control is only supported for ElRobot; "
                f"detected arm is {arm_state['arm_label']}"
            )

        joint_rad: dict[int, float] = {}
        for joint in arm_state["joints"]:
            motor_id = joint["motor_id"]
            if "present_position_normalized" not in joint:
                raise RuntimeError(
                    f"Motor {motor_id} is not calibrated (no range_min/range_max); "
                    "cannot compute gripper position"
                )
            joint_rad[motor_id] = kinematics.normalized_to_radians(
                joint["present_position_normalized"], motor_id
            )

        xyz = kinematics.forward_kinematics(joint_rad)
        return {
            "bus_serial": arm_state["bus_serial"],
            "xyz": list(xyz),
            "joint_rad": joint_rad,
        }

    async def pick_at_xyz(
        self,
        target_xyz: list[float],
        *,
        approach_height_m: float = 0.05,
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Pick up an object at a Cartesian point: open the gripper, approach from
        approach_height_m above the target, descend, close the gripper, then retreat
        back up to the approach height. ElRobot only. Raises immediately on any step's
        failure -- no partial-completion recovery.
        """
        await self.ensure_connected()
        await self.wait_for_inference()
        resolved_serial, _, profile = self._resolve_bus(bus_serial)
        self._require_elrobot(profile)

        x, y, z = (float(v) for v in target_xyz)
        above = [x, y, z + approach_height_m]

        steps = [
            ("open_gripper", await self.open_gripper(resolved_serial)),
            ("approach", await self.move_to_xyz(above, resolved_serial)),
            ("descend", await self.move_to_xyz([x, y, z], resolved_serial)),
            ("close_gripper", await self.close_gripper(resolved_serial)),
            ("retreat", await self.move_to_xyz(above, resolved_serial)),
        ]
        return {
            "target_xyz": [x, y, z],
            "approach_height_m": approach_height_m,
            "steps": [{"step": name, "result": result} for name, result in steps],
        }

    async def place_at_xyz(
        self,
        target_xyz: list[float],
        *,
        approach_height_m: float = 0.05,
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Place a held object at a Cartesian point: approach from approach_height_m
        above the target, descend, open the gripper to release, then retreat back up
        to the approach height. Assumes the gripper is already holding something (e.g.
        after pick_at_xyz). ElRobot only. Raises immediately on any step's failure --
        no partial-completion recovery.
        """
        await self.ensure_connected()
        await self.wait_for_inference()
        resolved_serial, _, profile = self._resolve_bus(bus_serial)
        self._require_elrobot(profile)

        x, y, z = (float(v) for v in target_xyz)
        above = [x, y, z + approach_height_m]

        steps = [
            ("approach", await self.move_to_xyz(above, resolved_serial)),
            ("descend", await self.move_to_xyz([x, y, z], resolved_serial)),
            ("open_gripper", await self.open_gripper(resolved_serial)),
            ("retreat", await self.move_to_xyz(above, resolved_serial)),
        ]
        return {
            "target_xyz": [x, y, z],
            "approach_height_m": approach_height_m,
            "steps": [{"step": name, "result": result} for name, result in steps],
        }

    async def go_home(self, bus_serial: str = "auto") -> dict[str, Any]:
        """Move all arm joints directly to kinematics.HOME_JOINT_RAD (the empirically
        safe, obstruction-free resting pose -- see kinematics.py's "Origin/home
        convention"). ElRobot only.

        Goes straight to joint-space via move_arm_pose rather than through move_to_xyz,
        since HOME_JOINT_RAD already *is* the joint angles -- routing it through IK
        would mean re-solving for a pose we already know exactly, for no benefit and
        with needless convergence risk.
        """
        await self.ensure_connected()
        await self.wait_for_inference()
        resolved_serial, _, profile = self._resolve_bus(bus_serial)
        self._require_elrobot(profile)

        normalized = kinematics.joint_rad_dict_to_normalized(kinematics.HOME_JOINT_RAD)
        result = await self.move_arm_pose(normalized, resolved_serial)
        result["pose"] = "home"
        return result

    async def move_through_waypoints(
        self,
        waypoints: list[list[float]],
        bus_serial: str = "auto",
    ) -> dict[str, Any]:
        """Move the gripper through a sequence of Cartesian points in order, via
        repeated move_to_xyz calls. ElRobot only. Raises immediately on any waypoint's
        failure (e.g. TESTING_MAX_MOVE_DELTA_M or MIN_SAFE_Z_M rejection, or IK
        non-convergence) -- no partial-completion recovery, matching pick_at_xyz/
        place_at_xyz. Earlier waypoints already reached are not undone.

        Covers circle/figure-eight traces and fixed multi-waypoint trajectories (e.g.
        pick zone -> transit -> drop zone) -- generate the point list and pass it in.
        For replaying a recorded joint-angle sequence from a file, read the file and
        call move_arm_pose per entry instead; this tool is XYZ-only.
        """
        await self.ensure_connected()
        await self.wait_for_inference()
        resolved_serial, _, profile = self._resolve_bus(bus_serial)
        self._require_elrobot(profile)

        if not waypoints:
            raise ValueError("waypoints must not be empty")

        steps = []
        for index, point in enumerate(waypoints):
            result = await self.move_to_xyz(point, resolved_serial)
            steps.append({"index": index, "target_xyz": [float(v) for v in point], "result": result})

        return {"waypoint_count": len(waypoints), "steps": steps}

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
