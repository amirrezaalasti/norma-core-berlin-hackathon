"""Ad-hoc manual test: read the gripper's current XYZ, then move it up by a small,
safe offset. Run with Station already up (--tcp) and the ElRobot connected:

    uv run --project software/station/mcp python software/station/mcp/test_small_move.py

Not part of the automated test suite (tests/test_kinematics.py) -- this one needs real
hardware and is meant to be watched while it runs.
"""

import asyncio

from norma_station_mcp.session import StationSession

# Lift straight up by this much (meters). Up is the safest single-axis offset to try
# first -- it can't drive the gripper into the table.
DELTA_Z = 0.03


async def main() -> None:
    session = StationSession("localhost:8888")
    await session.ensure_connected()
    await session.wait_for_inference(timeout_s=5.0)

    current = await session.get_gripper_xyz()
    x, y, z = current["xyz"]
    print("current xyz:", current["xyz"])

    target = [x, y, z + DELTA_Z]
    print("target xyz: ", target)

    result = await session.move_to_xyz(target)
    print("move result:", result)


if __name__ == "__main__":
    asyncio.run(main())
