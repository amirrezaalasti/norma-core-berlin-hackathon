"""In-process fake of station_py's Client/new_station_client/send_commands, for
exercising StationSession's control-flow logic with no real Station daemon or motors.

Activated via STATION_MOCK=1 (see session.py's conditional import) -- never imported
otherwise, so STATION_MOCK=0 users pay no cost and need no extra dependency.

Duck-types only what StationSession actually touches on `client`: `.connected`,
`.setup_done`, `.follow(queue_id, target) -> error_queue`. Frames are built with the
real `st3215.InferenceState` writer classes (same "gremlin" codegen used by real
Station) so `InferenceStateReader` parses them identically to real hardware frames --
no parallel/ad-hoc wire format.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import setup_import_paths

setup_import_paths()

from station_py.client import StreamEntry, StreamEntryId  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402

from . import kinematics  # noqa: E402
from .motor_state import (  # noqa: E402
    RAM_ACC,
    RAM_GOAL_POSITION,
    RAM_GOAL_SPEED,
    RAM_PRESENT_CURRENT,
    RAM_PRESENT_POSITION,
    RAM_STATUS,
    RAM_TORQUE_ENABLE,
    normalized_to_steps,
)

FAKE_BUS_SERIAL = "FAKE-BUS-01"
FAKE_FRAME_INTERVAL_S = 0.075

DEFAULT_RANGE_MIN = 100
DEFAULT_RANGE_MAX = 4000
MAX_RAW_STEP = 4095

GRIPPER_MOTOR_ID = 8
ELROBOT_MOTOR_IDS: tuple[int, ...] = tuple(range(1, 9))
GRIPPER_HOME_NORMALIZED = 0.0  # open at rest

STATE_FILE_PATH = Path(__file__).resolve().parent / ".fake_station_state.json"

# Written by scripted_motions_demo.py (or any other driver script), read by
# sim_visualizer.py to render an on-screen label -- a separate, optional channel from
# the motor-state file above, since "which named motion is currently running" is a
# concept that only exists in the driver script, not in the fake backend itself.
CURRENT_MOTION_LABEL_PATH = Path(__file__).resolve().parent / ".current_motion_label.txt"


def write_current_motion_label(text: str) -> None:
    tmp_path = CURRENT_MOTION_LABEL_PATH.with_suffix(".txt.tmp")
    tmp_path.write_text(text)
    for attempt in range(5):
        try:
            os.replace(tmp_path, CURRENT_MOTION_LABEL_PATH)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.005)


@dataclass
class FakeMotor:
    motor_id: int
    present_position: int
    range_min: int = DEFAULT_RANGE_MIN
    range_max: int = DEFAULT_RANGE_MAX
    torque_enabled: bool = False
    goal_speed: int = 0
    goal_accel: int = 0
    present_current: int = 0
    error_status: int = 0

    def normalized_position(self) -> float:
        span = self.range_max - self.range_min
        return (self.present_position - self.range_min) / span

    def encode_state(self) -> bytes:
        """Raw register blob matching motor_state.py's offsets. Goal == present always --
        the fake has no physical inertia/interpolation, only instant position snapping."""
        buf = bytearray(RAM_PRESENT_CURRENT + 2)
        buf[RAM_TORQUE_ENABLE] = 1 if self.torque_enabled else 0
        buf[RAM_ACC] = self.goal_accel & 0xFF
        buf[RAM_GOAL_POSITION : RAM_GOAL_POSITION + 2] = self.present_position.to_bytes(2, "little")
        buf[RAM_GOAL_SPEED : RAM_GOAL_SPEED + 2] = self.goal_speed.to_bytes(2, "little")
        buf[RAM_PRESENT_POSITION : RAM_PRESENT_POSITION + 2] = self.present_position.to_bytes(2, "little")
        buf[RAM_STATUS] = self.error_status & 0xFF
        buf[RAM_PRESENT_CURRENT : RAM_PRESENT_CURRENT + 2] = self.present_current.to_bytes(2, "little")
        return bytes(buf)


def _home_steps(motor_id: int, range_min: int, range_max: int) -> int:
    if motor_id == GRIPPER_MOTOR_ID:
        normalized = GRIPPER_HOME_NORMALIZED
    else:
        normalized = kinematics.radians_to_normalized(kinematics.HOME_JOINT_RAD[motor_id], motor_id)
    return normalized_to_steps(normalized, range_min, range_max)


class FakeBus:
    """Exactly one bus, fixed serial -- satisfies resolve_bus_serial's auto-mode (which
    requires exactly one bus). Motor ids 1-8 so detect_arm_profile recognizes ElRobot."""

    def __init__(self, motor_ids: tuple[int, ...] = ELROBOT_MOTOR_IDS, serial: str = FAKE_BUS_SERIAL):
        self.serial = serial
        self.motors: dict[int, FakeMotor] = {}
        for motor_id in motor_ids:
            steps = _home_steps(motor_id, DEFAULT_RANGE_MIN, DEFAULT_RANGE_MAX)
            self.motors[motor_id] = FakeMotor(motor_id=motor_id, present_position=steps)

    def encode_inference_state(self) -> bytes:
        return st3215.InferenceState(
            buses=[
                st3215.InferenceState_BusState(
                    bus=st3215.ST3215Bus(serial_number=self.serial),
                    motors=[
                        st3215.InferenceState_MotorState(
                            id=motor.motor_id,
                            state=motor.encode_state(),
                            range_min=motor.range_min,
                            range_max=motor.range_max,
                        )
                        for motor in self.motors.values()
                    ],
                )
            ]
        ).encode()

    def write_state_file(self) -> None:
        """Atomic write so sim_visualizer.py never reads a torn file. Retries
        os.replace on Windows -- unlike POSIX rename, MoveFileEx can transiently fail
        with PermissionError/WinError 5 if the visualizer's concurrent read of the
        destination path overlaps the instant of replacement; the conflicting handle is
        only ever held for the duration of a single read, so a few short retries are
        sufficient (observed in practice: a visualizer polling at 50ms racing a writer
        at 75ms)."""
        payload: dict[str, Any] = {
            str(motor_id): motor.normalized_position() for motor_id, motor in self.motors.items()
        }
        tmp_path = STATE_FILE_PATH.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload))
        for attempt in range(5):
            try:
                os.replace(tmp_path, STATE_FILE_PATH)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.005)


class FakeClient:
    def __init__(self, logger):
        self.logger = logger
        self.connected = True
        self.setup_done = True
        self.bus = FakeBus()
        self._target_queue: asyncio.Queue | None = None
        self._follow_task: asyncio.Task | None = None
        self._seq = 0

    def follow(self, queue_id: str, target: asyncio.Queue) -> asyncio.Queue:
        error_queue: asyncio.Queue = asyncio.Queue()
        if queue_id != "st3215/inference":
            error_queue.put_nowait(RuntimeError(f"FakeClient cannot follow {queue_id!r}"))
            return error_queue
        self._target_queue = target
        self._follow_task = asyncio.create_task(self._pump())
        return error_queue

    async def _pump(self) -> None:
        """Periodic, best-effort refresh -- unlike push_frame_now's synchronous callers
        (send_commands, where a fresh frame is required for read-after-write
        consistency), a single missed periodic tick is harmless: the next one follows
        in FAKE_FRAME_INTERVAL_S regardless. So a transient PermissionError here (the
        Windows file-race write_state_file already retries, but isn't guaranteed to
        win the race within its retry budget) is swallowed rather than left as an
        unhandled task exception."""
        while True:
            try:
                self.push_frame_now()
            except PermissionError:
                pass
            await asyncio.sleep(FAKE_FRAME_INTERVAL_S)

    def push_frame_now(self) -> None:
        """Synchronous re-emit. send_commands calls this right after applying writes,
        then yields once (asyncio.sleep(0)) so the StationSession's _follow_inference
        consumer task drains it before the next composite-call step reads
        latest_inference -- on real hardware this ordering falls out for free because
        send_commands does genuine socket I/O (an inherent yield point); the in-memory
        fake has no such I/O, so the yield has to be forced explicitly here."""
        if self._target_queue is None:
            return
        self._seq += 1
        entry = StreamEntry(
            ID=StreamEntryId(ID=str(self._seq).encode()),
            Data=memoryview(self.bus.encode_inference_state()),
        )
        self._target_queue.put_nowait(entry)
        self.bus.write_state_file()


async def new_station_client(server: str, logger) -> FakeClient:
    logger.info("STATION_MOCK=1: using in-process FakeClient (no real station/hardware)")
    return FakeClient(logger)


async def send_commands(client: FakeClient, command_list: list) -> None:
    for cmd in command_list:
        _apply_command(client.bus, cmd)
    client.push_frame_now()
    # Force a real yield so StationSession's _follow_inference consumer task (already
    # woken by the put_nowait above) actually drains the fresh frame into
    # latest_inference before this coroutine returns -- see push_frame_now's docstring.
    await asyncio.sleep(0)


def _apply_command(bus: FakeBus, cmd: Any) -> None:
    reader = st3215.CommandReader(cmd.body)

    # get_sync_write()/get_write() never return None (gremlin readers default to an
    # empty-buffer reader, not None) -- presence must be checked via parsed content, not
    # an `is not None` check.
    sync_write = reader.get_sync_write()
    motors = sync_write.get_motors() or []
    if motors:
        address = sync_write.get_address()
        for motor_write in motors:
            _apply_write(bus, motor_write.get_motor_id(), address, bytes(motor_write.get_value()))
        return

    write = reader.get_write()
    if write.get_address() != 0:
        _apply_write(bus, write.get_motor_id(), write.get_address(), bytes(write.get_value()))


def _apply_write(bus: FakeBus, motor_id: int, address: int, value: bytes) -> None:
    motor = bus.motors.get(motor_id)
    if motor is None:
        return
    if address == RAM_TORQUE_ENABLE:
        motor.torque_enabled = bool(value) and value[0] != 0
    elif address == RAM_GOAL_POSITION:
        if len(value) >= 2:
            position = int.from_bytes(value[:2], "little")
            motor.present_position = max(0, min(MAX_RAW_STEP, position))
