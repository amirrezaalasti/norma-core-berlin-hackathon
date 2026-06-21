"""Rehearses the kinematics-checklist's section-3 scripted-motion repertoire end to
end, against either the real station or (typically) the STATION_MOCK=1 fake. Run
alongside sim_visualizer.py to watch each motion:

    STATION_MOCK=1 python -m norma_station_mcp.scripted_motions_demo

Every motion here is composed entirely from existing tools (move_joint, move_arm_pose,
open_gripper/close_gripper) plus the two new primitives (go_home,
move_through_waypoints) -- no motion-specific server code exists beyond those two.
ElRobot only, same as the Cartesian tools it builds on.

This is a visual/manual rehearsal aid, not a pass/fail test suite -- correctness of the
underlying primitives is covered by tests/test_session_mock.py. Running clean here
confirms the *simulated* logic and IK math are self-consistent; it does not confirm
real hardware will move the same way (see kinematics.py's calibration caveat).
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

from . import kinematics
from .fake_station import write_current_motion_label
from .session import StationSession

WRIST_JOINT_ID = 7
NOD_JOINT_ID = 2
OSCILLATION_DELTAS = (0.05, -0.05, 0.05, -0.05, 0.0)
STEP_PAUSE_S = 0.3
SWEEP_SUB_STEPS = 8
SWEEP_SUB_STEP_PAUSE_S = 0.05
EXAMPLE_SEQUENCE_PATH = Path(__file__).resolve().parent / "example_joint_sequence.json"


async def _pause() -> None:
    await asyncio.sleep(STEP_PAUSE_S)


def _current_normalized(session: StationSession, joint_id: int) -> float:
    motor = session.get_motor("auto", joint_id)
    return (motor["present_position"] - motor["range_min"]) / (motor["range_max"] - motor["range_min"])


async def _sweep_to(session: StationSession, joint_id: int, target_position: float) -> None:
    """Interpolates from the joint's current position to target_position in small
    sub-steps, so the motion reads as an actual sweep instead of snapping directly
    between far-apart extremes. If a sub-step is rejected (workspace floor), stops the
    sweep toward this target right there instead of skipping ahead to the next target --
    that's the safety check correctly stopping at the edge of the danger zone, not
    jumping past it.
    """
    start = _current_normalized(session, joint_id)
    for i in range(1, SWEEP_SUB_STEPS + 1):
        position = start + (target_position - start) * i / SWEEP_SUB_STEPS
        try:
            await session.move_joint(joint_id, position)
        except RuntimeError as exc:
            print(f"   joint {joint_id} @ {position:.2f}: rejected, stopping sweep toward {target_position} -- {exc}")
            return
        await asyncio.sleep(SWEEP_SUB_STEP_PAUSE_S)


def _label(text: str) -> None:
    """Prints AND writes to the file sim_visualizer.py renders as on-screen text --
    print() alone is invisible to anyone just watching the PyBullet window, since it
    only shows up in whichever terminal happens to be running this script."""
    print(f"-- {text} --")
    write_current_motion_label(text)


def _oscillation_positions(joint_id: int, deltas: tuple[float, ...] = OSCILLATION_DELTAS) -> list[float]:
    """Small swing relative to this joint's actual home position, clamped to [0, 1].

    Earlier versions of wave/nod used hardcoded absolute targets (0.4-0.6) regardless
    of where home actually sits for that joint. For joint 2 home is normalized ~0.011 --
    near one extreme of its range -- so a "0.4-0.6" swing was a huge jump away from
    home, not a gentle oscillation, and dragged the arm down toward the table on real
    forward kinematics. Offsetting from the joint's own home value keeps this a genuine
    small wave/nod regardless of where that joint's home happens to sit.
    """
    home_normalized = kinematics.radians_to_normalized(kinematics.HOME_JOINT_RAD[joint_id], joint_id)
    return [max(0.0, min(1.0, home_normalized + delta)) for delta in deltas]


async def wave(session: StationSession) -> None:
    _label("1. wave: oscillating wrist joint")
    for position in _oscillation_positions(WRIST_JOINT_ID):
        await session.move_joint(WRIST_JOINT_ID, position)
        await _pause()


async def move_to_home(session: StationSession) -> None:
    _label("2. home: returning to safe resting pose")
    await session.go_home()
    await _pause()


async def squeeze_gripper(session: StationSession) -> None:
    _label("3. gripper squeeze: open/close repeatedly")
    for _ in range(3):
        await session.close_gripper()
        await _pause()
        await session.open_gripper()
        await _pause()


def _circle_waypoints(radius_m: float = 0.02, points: int = 8) -> list[list[float]]:
    return [
        [radius_m * math.cos(2 * math.pi * i / points), radius_m * math.sin(2 * math.pi * i / points), 0.0]
        for i in range(points)
    ]


async def trace_circle(session: StationSession) -> None:
    _label("4. circle trace")
    for point in _circle_waypoints():
        await session.move_to_xyz(point)
        await _pause()


async def taught_point(session: StationSession) -> None:
    _label("5. taught point: hover over drop zone")
    drop_zone = [0.02, -0.02, 0.01]
    await session.move_to_xyz(drop_zone)
    await _pause()


async def sweep_joints(session: StationSession) -> None:
    """Deliberately sweeps the full normalized range -- this is expected to legitimately
    trip MIN_SAFE_Z_M for some joint/position combinations (that's the workspace floor
    protecting the table, working as intended, not a bug). Interpolates each leg via
    _sweep_to rather than snapping directly to (0.1, 0.9, 0.5), so it visibly reads as
    a sweep; a rejected sub-step stops that leg early instead of skipping ahead.
    """
    _label("6. sweep: each joint across its full range, one at a time")
    for joint_id in range(1, 8):
        for position in (0.1, 0.9, 0.5):
            await _sweep_to(session, joint_id, position)
    await session.go_home()


async def nod(session: StationSession) -> None:
    _label("7. nod: oscillating shoulder joint")
    for position in _oscillation_positions(NOD_JOINT_ID):
        await session.move_joint(NOD_JOINT_ID, position)
        await _pause()


async def safe_return(session: StationSession) -> None:
    _label("8. safe return: home before shutdown")
    await session.go_home()
    await _pause()


async def pick_transit_drop(session: StationSession) -> None:
    _label("9. multi-waypoint trajectory: pick zone -> transit -> drop zone")
    pick_zone = [0.02, 0.0, 0.0]
    transit = [0.0, 0.0, 0.02]
    drop_zone = [-0.02, 0.0, 0.0]
    for point in (pick_zone, transit, drop_zone):
        await session.move_to_xyz(point)
        await _pause()


async def replay_recorded_sequence(session: StationSession) -> None:
    _label(f"10. replay: recorded joint-angle sequence from {EXAMPLE_SEQUENCE_PATH.name}")
    poses = json.loads(EXAMPLE_SEQUENCE_PATH.read_text())
    for pose in poses:
        await session.move_arm_pose({int(joint_id): position for joint_id, position in pose.items()})
        await _pause()


async def main() -> None:
    session = StationSession()
    await session.wait_for_inference()

    # move_joint-based segments (wave, sweep, nod) can leave the gripper anywhere in
    # Cartesian space -- go_home is joint-space and always safe regardless of current
    # position, so it's used as a reset before every Cartesian (move_to_xyz-based)
    # segment to keep this script's waypoints valid relative to a known starting point.
    await move_to_home(session)
    await wave(session)
    await squeeze_gripper(session)
    await move_to_home(session)
    await trace_circle(session)
    await move_to_home(session)
    await taught_point(session)
    await sweep_joints(session)
    await nod(session)
    await move_to_home(session)
    await pick_transit_drop(session)
    await replay_recorded_sequence(session)
    await safe_return(session)

    print("done.")


if __name__ == "__main__":
    asyncio.run(main())
