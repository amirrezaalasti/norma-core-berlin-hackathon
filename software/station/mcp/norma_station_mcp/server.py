from __future__ import annotations

import json
import logging

from fastmcp import FastMCP

from .session import get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("norma-station-mcp")

mcp = FastMCP(
    name="NormaCore Station",
    instructions=(
        "NormaCore robotics MCP server. Station must run with `--tcp` (port 8888).\n\n"
        "Prefer high-level tools:\n"
        "- get_arm_state: read joints + gripper with roles\n"
        "- move_joint / move_arm_pose: joint-space motion (0.0-1.0 per joint)\n"
        "- open_gripper / close_gripper / set_gripper: gripper control\n"
        "- enable_arm_torque / disable_arm_torque: power all motors\n"
        "- move_gripper_to_xyz / pick_at_xyz / place_at_xyz: Cartesian (XYZ, meters, "
        "base_link frame) control via inverse kinematics -- ElRobot only. "
        "get_gripper_xyz reads current XYZ.\n\n"
        "Joint ids match motor ids (SO-101: joints 1-5, gripper 6; ElRobot: joints 1-7, gripper 8).\n"
        "Joint-space positions are normalized 0.0-1.0 per motor; Cartesian tools take XYZ meters instead.\n"
        "Low-level advanced_* tools exist for debugging."
    ),
)


def _json(data: object) -> str:
    return json.dumps(data, indent=2)


# ── Discovery & state ─────────────────────────────────────────────────────────


@mcp.tool
async def station_connection_status() -> str:
    """Check connectivity to the NormaCore Station TCP server."""
    session = get_session()
    try:
        await session.ensure_connected()
        await session.wait_for_inference(timeout_s=5.0)
    except Exception as exc:
        info = session.connection_info()
        info["error"] = str(exc)
        return _json(info)

    info = session.connection_info()
    info["bus_count"] = len(session.list_buses())
    return _json(info)


@mcp.tool
async def get_arm_state(bus_serial: str = "auto") -> str:
    """Read the full arm: detected type (SO-101 / ElRobot), joint positions, and gripper state.

    Start here before moving the robot. Returns normalized positions (0.0-1.0) per joint.
    """
    session = get_session()
    await session.ensure_connected()
    await session.wait_for_inference()
    return _json(session.get_arm_state(bus_serial))


# ── Gripper ───────────────────────────────────────────────────────────────────


@mcp.tool
async def open_gripper(bus_serial: str = "auto") -> str:
    """Fully open the gripper (position 0.0 on the gripper motor's calibrated range)."""
    session = get_session()
    return _json(await session.open_gripper(bus_serial))


@mcp.tool
async def close_gripper(bus_serial: str = "auto") -> str:
    """Fully close the gripper (position 1.0 on the gripper motor's calibrated range)."""
    session = get_session()
    return _json(await session.close_gripper(bus_serial))


@mcp.tool
async def set_gripper(position: float, bus_serial: str = "auto") -> str:
    """Set gripper opening. 0.0 = open, 1.0 = closed, values in between = partial grasp."""
    session = get_session()
    return _json(await session.set_gripper(position, bus_serial))


# ── Arm motion (joint space) ──────────────────────────────────────────────────


@mcp.tool
async def move_joint(joint_id: int, position: float, bus_serial: str = "auto") -> str:
    """Move one arm joint to a normalized position (0.0 = min, 1.0 = max).

    Joint id equals motor id: SO-101 joints are 1-5, ElRobot joints are 1-7.
    Does not move the gripper — use open_gripper / close_gripper for that.
    """
    session = get_session()
    return _json(await session.move_joint(joint_id, position, bus_serial))


@mcp.tool
async def move_arm_pose(
    joint_positions: dict[int, float],
    bus_serial: str = "auto",
) -> str:
    """Move multiple arm joints at once. Example: {1: 0.5, 2: 0.3, 3: 0.8}.

    Keys are joint ids (motor ids). Values are normalized 0.0-1.0 within each
    joint's calibrated range. Gripper is not included — control it separately.
    """
    session = get_session()
    return _json(await session.move_arm_pose(joint_positions, bus_serial))


@mcp.tool
async def enable_arm_torque(bus_serial: str = "auto") -> str:
    """Enable torque on all motors (required before the arm can hold position)."""
    session = get_session()
    return _json(await session.enable_arm_torque(bus_serial))


@mcp.tool
async def disable_arm_torque(bus_serial: str = "auto") -> str:
    """Disable torque on all motors (arm goes limp — use with care)."""
    session = get_session()
    return _json(await session.disable_arm_torque(bus_serial))


# ── Arm motion (Cartesian / IK) ────────────────────────────────────────────────


@mcp.tool
async def move_gripper_to_xyz(target_xyz: list[float], bus_serial: str = "auto") -> str:
    """Move the gripper to a Cartesian point [x, y, z] in meters using inverse kinematics.

    ElRobot only (7-DOF arm). Coordinates are in the robot's base_link frame -- if you
    got this point from a camera/vision system, it must already be transformed into
    base_link coordinates (hand-eye calibration is not handled by this server). Solves
    IK and moves all 7 arm joints in one motion; does not touch the gripper open/close
    state.
    """
    session = get_session()
    return _json(await session.move_to_xyz(target_xyz, bus_serial))


@mcp.tool
async def get_gripper_xyz(bus_serial: str = "auto") -> str:
    """Read the gripper's current Cartesian position [x, y, z] in meters (base_link frame).

    Computed via forward kinematics from the arm's current joint positions. ElRobot only.
    """
    session = get_session()
    return _json(await session.get_gripper_xyz(bus_serial))


@mcp.tool
async def pick_at_xyz(
    target_xyz: list[float],
    approach_height_m: float = 0.05,
    bus_serial: str = "auto",
) -> str:
    """Pick up an object at Cartesian point [x, y, z] (meters, base_link frame).

    Composite motion: open gripper, approach from approach_height_m above the target,
    descend to the target, close the gripper, then retreat back up to the approach
    height. ElRobot only. Use get_gripper_xyz / move_gripper_to_xyz for finer-grained
    control.
    """
    session = get_session()
    return _json(
        await session.pick_at_xyz(
            target_xyz, approach_height_m=approach_height_m, bus_serial=bus_serial
        )
    )


@mcp.tool
async def place_at_xyz(
    target_xyz: list[float],
    approach_height_m: float = 0.05,
    bus_serial: str = "auto",
) -> str:
    """Place a held object at Cartesian point [x, y, z] (meters, base_link frame).

    Composite motion: approach from approach_height_m above the target, descend to the
    target, open the gripper to release, then retreat back up to the approach height.
    Assumes the gripper is already holding something (e.g. after pick_at_xyz). ElRobot
    only.
    """
    session = get_session()
    return _json(
        await session.place_at_xyz(
            target_xyz, approach_height_m=approach_height_m, bus_serial=bus_serial
        )
    )


# ── Advanced / low-level ─────────────────────────────────────────────────────


@mcp.tool
async def advanced_list_motor_buses() -> str:
    """Low-level: raw ST3215 bus list with unlabeled motor registers."""
    session = get_session()
    await session.ensure_connected()
    await session.wait_for_inference()
    return _json({"buses": session.list_buses()})


@mcp.tool
async def advanced_get_motor_state(bus_serial: str = "auto", motor_id: int = 1) -> str:
    """Low-level: read one motor by id without arm role labels."""
    session = get_session()
    await session.ensure_connected()
    await session.wait_for_inference()
    return _json(session.get_motor(bus_serial, motor_id))


@mcp.tool
async def advanced_move_motor_normalized(
    motor_id: int,
    position: float,
    bus_serial: str = "auto",
) -> str:
    """Low-level: move any motor by id to normalized position 0.0-1.0."""
    session = get_session()
    return _json(await session.move_motor_normalized(motor_id, position, bus_serial))


@mcp.tool
async def advanced_move_motor_steps(
    motor_id: int,
    goal_steps: int,
    bus_serial: str = "auto",
) -> str:
    """Low-level: move any motor to absolute encoder steps."""
    session = get_session()
    return _json(await session.move_motor_steps(motor_id, goal_steps, bus_serial))


@mcp.tool
async def advanced_set_motor_torque(
    motor_ids: list[int],
    enable: bool,
    bus_serial: str = "auto",
) -> str:
    """Low-level: enable/disable torque on specific motor ids."""
    session = get_session()
    return _json(await session.set_torque(motor_ids, enable, bus_serial))


def main() -> None:
    logger.info("Starting NormaCore Station MCP server (stdio)")
    mcp.run()


if __name__ == "__main__":
    main()
