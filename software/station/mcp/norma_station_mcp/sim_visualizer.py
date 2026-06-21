"""Standalone PyBullet viewer for the STATION_MOCK=1 fake arm. Run in a second terminal
alongside the MCP server:

    python -m norma_station_mcp.sim_visualizer

Polls fake_station.py's state hand-off file and mirrors the simulated arm's pose into a
PyBullet GUI window. Deliberately a separate process, not a thread inside the MCP
server -- PyBullet's GUI needs to own its process's main thread, and isolating it means
a visualizer crash can't take down the live MCP stdio connection.

Requires the `sim` extra (`pip install -e ".[sim]"` or `uv sync --extra sim`); never
imported by server.py/session.py, so STATION_MOCK=0 users don't need pybullet installed.
"""

from __future__ import annotations

import json
import time

import pybullet as p

from .fake_station import CURRENT_MOTION_LABEL_PATH, ELROBOT_MOTOR_IDS, GRIPPER_MOTOR_ID, STATE_FILE_PATH
from .kinematics import ELROBOT_URDF_PATH, normalized_to_radians

LABEL_POSITION = (0.0, 0.0, 0.35)
LABEL_COLOR = (1.0, 1.0, 1.0)
LABEL_SIZE = 1.5

# Standard RGB = XYZ convention. Anchored at AXIS_ORIGIN, in PyBullet's raw world frame
# (where useFixedBase placed the robot's base_link) -- NOT kinematics.py's home-relative
# origin (gripper-at-home = (0,0,0)), which sits ~0.11m higher in Z and offset in X/Y.
AXIS_ORIGIN = (0.0, 0.0, 0.0)
AXIS_LENGTH_M = 1.0
WORLD_AXES = (
    ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # X, red
    ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),  # Y, green
    ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),  # Z, blue
)


def _draw_world_axes() -> None:
    for direction, color in WORLD_AXES:
        endpoint = tuple(o + AXIS_LENGTH_M * d for o, d in zip(AXIS_ORIGIN, direction))
        p.addUserDebugLine(AXIS_ORIGIN, endpoint, lineColorRGB=color, lineWidth=2)

POLL_INTERVAL_S = 0.05
ARM_JOINT_IDS = tuple(i for i in ELROBOT_MOTOR_IDS if i != GRIPPER_MOTOR_ID)

# The gripper's two jaw fingers are separate prismatic joints in the URDF, mechanically
# linked to the main gripper joint (rev_motor_08) via <mimic multiplier=... offset="0">.
# PyBullet does not evaluate URDF <mimic> tags automatically, so they must be driven
# explicitly here or they sit frozen at 0 while the gripper body moves through them.
GRIPPER_FINGER_MIMICS = {
    "rev_motor_08_1": -0.0115,
    "rev_motor_08_2": 0.0115,
}


def _joint_name(motor_id: int) -> str:
    return f"rev_motor_0{motor_id}"


def _build_joint_index_map(body: int) -> tuple[dict[int, int], dict[str, int]]:
    name_to_index = {}
    for i in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, i)
        name_to_index[info[1].decode()] = i

    joint_index_by_motor: dict[int, int] = {}
    for motor_id in ARM_JOINT_IDS:
        name = _joint_name(motor_id)
        if name in name_to_index:
            joint_index_by_motor[motor_id] = name_to_index[name]
        else:
            print(f"warning: no URDF joint named {name!r} -- motor {motor_id} won't be driven")

    gripper_name = _joint_name(GRIPPER_MOTOR_ID)
    if gripper_name in name_to_index:
        joint_index_by_motor[GRIPPER_MOTOR_ID] = name_to_index[gripper_name]
    else:
        print(f"warning: no URDF joint named {gripper_name!r} -- gripper won't be driven")

    finger_joint_indices: dict[str, int] = {}
    for joint_name in GRIPPER_FINGER_MIMICS:
        if joint_name in name_to_index:
            finger_joint_indices[joint_name] = name_to_index[joint_name]
        else:
            print(f"warning: no URDF joint named {joint_name!r} -- gripper finger won't be driven")

    return joint_index_by_motor, finger_joint_indices


def _gripper_angle(body: int, joint_index: int, normalized: float) -> float:
    info = p.getJointInfo(body, joint_index)
    lower, upper = info[8], info[9]
    return lower + (upper - lower) * normalized


def _read_state() -> dict[str, float] | None:
    try:
        return json.loads(STATE_FILE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        # PermissionError happens when this read overlaps fake_station.py's os.replace
        # of the same path on Windows -- transient, just skip this tick and retry on
        # the next poll.
        return None


def _read_label() -> str | None:
    try:
        return CURRENT_MOTION_LABEL_PATH.read_text()
    except (FileNotFoundError, PermissionError):
        return None


def main() -> None:
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)
    body = p.loadURDF(str(ELROBOT_URDF_PATH), useFixedBase=True)
    joint_index_by_motor, finger_joint_indices = _build_joint_index_map(body)
    _draw_world_axes()

    print(f"watching {STATE_FILE_PATH} -- start the MCP server with STATION_MOCK=1 "
          "and call its tools to see the arm move.")

    last_label: str | None = None
    label_item_id: int | None = None

    try:
        while True:
            state = _read_state()
            if state is not None:
                for motor_id_str, normalized in state.items():
                    motor_id = int(motor_id_str)
                    joint_index = joint_index_by_motor.get(motor_id)
                    if joint_index is None:
                        continue
                    if motor_id == GRIPPER_MOTOR_ID:
                        angle = _gripper_angle(body, joint_index, normalized)
                        for joint_name, multiplier in GRIPPER_FINGER_MIMICS.items():
                            finger_index = finger_joint_indices.get(joint_name)
                            if finger_index is not None:
                                p.resetJointState(body, finger_index, multiplier * angle)
                    else:
                        angle = normalized_to_radians(normalized, motor_id)
                    p.resetJointState(body, joint_index, angle)

            label = _read_label()
            if label is not None and label != last_label:
                kwargs = {"replaceItemUniqueId": label_item_id} if label_item_id is not None else {}
                label_item_id = p.addUserDebugText(
                    label, LABEL_POSITION, textColorRGB=LABEL_COLOR, textSize=LABEL_SIZE, **kwargs
                )
                last_label = label

            p.stepSimulation()
            time.sleep(POLL_INTERVAL_S)
    except p.error:
        # Raised once resetJointState/stepSimulation runs after the GUI window has been
        # closed (the physics-server connection is torn down with it) -- not a bug,
        # just this process's natural exit signal since there's no other "window
        # closed" callback exposed for GUI mode.
        print("window closed -- exiting.")


if __name__ == "__main__":
    main()
