from __future__ import annotations

import json
import logging

from fastmcp import FastMCP

from .session import get_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("norma-station-mcp")


def _json(data: object) -> str:
    return json.dumps(data, indent=2)


mcp = FastMCP(
    name="NormaCore Station",
    instructions=(
        "NormaCore robotics MCP server. Station must run with `--tcp` (port 8888).\n\n"
        "Prefer high-level pick/place tools:\n"
        "- go_home: move to saved home pose (optionally open gripper)\n"
        "- move_direction: nudge up/down/left/right using teleop-calibrated joint deltas\n"
        "- pick_object: open gripper, move to static pick pose, close gripper\n"
        "- lift_object: move to home while holding object (gripper stays closed)\n"
        "- place_object: move to static pick pose, open gripper, return home\n"
        "- get_fixed_pick_pose / get_home_pose: read home and static pick poses\n\n"
        "Pick/placement always uses hardcoded STATIC_PICK_JOINTS — no vision offset planning.\n\n"
        "Other tools:\n"
        "- get_arm_state: read joints + gripper with roles\n"
        "- move_direction: calibrated up/down/left/right nudge (amount=1.0 is one teleop step)\n"
        "- move_joint / move_arm_pose: joint-space motion (0.0-1.0 per joint)\n"
        "- open_gripper / close_gripper / set_gripper: gripper control\n"
        "- enable_arm_torque / disable_arm_torque: power all motors\n"
        "- detect_objects / detect_workspace_objects: optional vision (not used for pick)\n"
        "- save_home_pose: save home pose\n"
        "- pick_nearest_object: alias for pick_object\n\n"
        "Joint ids match motor ids (SO-101: joints 1-5, gripper 6; ElRobot: joints 1-7, gripper 8).\n"
        "Positions are normalized within each motor's calibrated range, not Cartesian XYZ.\n"
        "Low-level advanced_* tools exist for debugging."
    ),
)


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
    """Fully open the gripper (position 1.0 on the gripper motor's calibrated range)."""
    session = get_session()
    return _json(await session.open_gripper(bus_serial))


@mcp.tool
async def close_gripper(bus_serial: str = "auto") -> str:
    """Fully close the gripper (position 0.0 on the gripper motor's calibrated range)."""
    session = get_session()
    return _json(await session.close_gripper(bus_serial))


@mcp.tool
async def set_gripper(position: float, bus_serial: str = "auto") -> str:
    """Set gripper opening. 0.0 = closed, 1.0 = open, values in between = partial grasp."""
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
async def move_direction(
    direction: str,
    amount: float = 1.0,
    bus_serial: str = "auto",
) -> str:
    """Nudge the arm up, down, left, or right using teleop-calibrated joint deltas.

    Applies coordinated joint changes from the current pose (not Cartesian mm).
    amount=1.0 is one teleop button press; use 2.0 for a double nudge.
    Prefer this over move_joint for voice commands like "go right" or "move up".
    """
    from .direction_control import move_direction as _move_direction

    session = get_session()
    return _json(
        await _move_direction(
            session,
            direction,
            amount=amount,
            bus_serial=bus_serial,
        )
    )


@mcp.tool
async def get_direction_calibration() -> str:
    """Return teleop direction nudge calibration (joint deltas per up/down/left/right)."""
    from .direction_control import DIRECTION_NUDGE_PATH, load_direction_nudge

    payload = load_direction_nudge()
    if payload is None:
        return _json(
            {
                "saved": False,
                "path": str(DIRECTION_NUDGE_PATH),
                "note": "Using built-in ElRobot defaults when file is missing.",
            }
        )
    return _json({"saved": True, **payload})


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


@mcp.tool
async def detect_workspace_objects(
    camera_index: int = 0,
    classes: list[str] | None = None,
) -> str:
    """Detect objects on the white board with gripper-relative offset and distance.

    Requires manual calibration in the station viewer (4 board points + gripper tip)
    synced to the vision server, with the arm at the home pose when setting the tip.
    Returns offset_xy and distance for each detection.
    Install vision deps: uv sync --project software/station/mcp --extra vision
    """
    from .vision_bridge import detect_workspace_objects as _detect

    payload = await _detect(camera_index=camera_index, classes=classes)
    return _json(payload)


@mcp.tool
async def get_fixed_pick_pose() -> str:
    """Return the static pick joint pose (hardcoded STATIC_PICK_JOINTS, never vision-derived)."""
    from .pick_control import STATIC_PICK_JOINTS

    return _json({"planning_mode": "static", "joint_positions": dict(STATIC_PICK_JOINTS)})


@mcp.tool
async def go_home(open_gripper: bool = True, bus_serial: str = "auto") -> str:
    """Move the arm to the saved home pose.

    Set open_gripper=false when returning home while holding an object (use lift_object instead).
    """
    from .pick_control import go_home as _go_home

    session = get_session()
    return _json(await _go_home(session, bus_serial=bus_serial, open_gripper=open_gripper))


@mcp.tool
async def pick_object(
    lift_after: bool = False,
    bus_serial: str = "auto",
) -> str:
    """Pick the object at the static pick pose.

    Sequence: home → open gripper → move to pick pose → wait until settled → close gripper.
    Set lift_after=true to move to home afterward while holding the object.
    """
    from .pick_control import pick_object as _pick_object

    session = get_session()
    return _json(
        await _pick_object(session, bus_serial=bus_serial, lift_after=lift_after)
    )


@mcp.tool
async def lift_object(bus_serial: str = "auto") -> str:
    """Lift a held object by moving to home without opening the gripper."""
    from .pick_control import lift_object as _lift_object

    session = get_session()
    return _json(await _lift_object(session, bus_serial=bus_serial))


@mcp.tool
async def place_object(bus_serial: str = "auto") -> str:
    """Place a held object back at the pick pose, open gripper, then return home."""
    from .pick_control import place_object as _place_object

    session = get_session()
    return _json(await _place_object(session, bus_serial=bus_serial))


@mcp.tool
async def save_home_pose(bus_serial: str = "auto") -> str:
    """Save the current arm joint positions as the initialized home pose.

    Move the arm to the pick-ready pose first, then calibrate the gripper tip in the
    station viewer while the arm is at this pose. All pick motions are deltas from home.
    """
    from .pick_control import home_pose_from_arm_state, save_home_pose as _save

    session = get_session()
    await session.ensure_connected()
    await session.wait_for_inference()
    arm_state = session.get_arm_state(bus_serial)
    payload = home_pose_from_arm_state(arm_state)
    return _json(_save(payload))


@mcp.tool
async def get_home_pose() -> str:
    """Return the saved initialized home pose, if any."""
    from .pick_control import load_home_pose

    payload = load_home_pose()
    if payload is None:
        return _json({"saved": False, "note": "Call save_home_pose with arm at init pose."})
    return _json({"saved": True, **payload})


@mcp.tool
async def get_pick_calibration() -> str:
    """Return empirical pick calibration (home→pick joint mapping vs vision offset in mm)."""
    from .pick_calibration import load_pick_calibration

    payload = load_pick_calibration()
    if payload is None:
        return _json(
            {
                "saved": False,
                "note": (
                    "No pick calibration saved. At home pose call save_home_pose, "
                    "move to a successful pick pose, then save_pick_reference with the "
                    "object board_xy from the camera overlay."
                ),
            }
        )
    return _json({"saved": True, **payload})


@mcp.tool
async def save_pick_reference(
    board_x: float,
    board_y: float,
    bus_serial: str = "auto",
) -> str:
    """Record a pick pose sample for offline calibration (not used by pick_object).

    pick_object / place_object always use hardcoded STATIC_PICK_JOINTS.
    """
    from .pick_control import save_pick_reference as _save

    session = get_session()
    payload = await _save(session, board_x=board_x, board_y=board_y, bus_serial=bus_serial)
    return _json({"saved": True, **payload})


@mcp.tool
async def pick_nearest_object(
    bus_serial: str = "auto",
    settle_s: float = 1.5,
    return_home: bool = False,
) -> str:
    """Pick at the static pick pose (alias for pick_object, no vision planning).

    When return_home=true, lifts to home after grasping (gripper stays closed).
    """
    from .pick_control import pick_nearest_object as _pick

    session = get_session()
    payload = await _pick(
        session,
        bus_serial=bus_serial,
        settle_s=settle_s,
        return_home=return_home,
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
