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
        "- detect_objects: pretrained YOLOE detection (requires vision extra)\n\n"
        "Joint ids match motor ids (SO-101: joints 1-5, gripper 6; ElRobot: joints 1-7, gripper 8).\n"
        "Positions are normalized within each motor's calibrated range, not Cartesian XYZ.\n"
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


# ── Vision (pretrained YOLOE / YOLO) ────────────────────────────────────────


@mcp.tool
async def detect_objects(
    classes: list[str] | None = None,
    camera_index: int = 0,
    confidence: float = 0.25,
) -> str:
    """Detect objects in the latest station camera frame using a pretrained model.

    No custom training — uses YOLOE with text prompts (default model: yoloe-11s-seg.pt).
    Returns pixel coordinates and oriented boxes [x, y, w, h, angle_deg] when masks
    are available. Install vision deps: uv sync --project software/station/mcp --extra vision

    Example classes: ["cube", "mug", "rectangular box"]
    """
    from .vision_bridge import DEFAULT_CLASSES, detect_from_station

    requested = classes or DEFAULT_CLASSES
    payload = await detect_from_station(
        classes=requested,
        camera_index=camera_index,
        confidence=confidence,
    )
    return _json(payload)


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
