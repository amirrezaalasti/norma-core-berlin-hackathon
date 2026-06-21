"""Integration tests for StationSession's control-flow logic against fake_station's
in-process fake -- no real Station daemon or motors needed. Complements
test_kinematics.py (pure IK/FK math) by covering the session/protobuf/command layer.
"""

from __future__ import annotations

import asyncio

import pytest

from norma_station_mcp import fake_station, kinematics
from norma_station_mcp import session as session_mod

ARM_JOINT_IDS = tuple(range(1, 8))
GRIPPER_ID = fake_station.GRIPPER_MOTOR_ID


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(session_mod, "new_station_client", fake_station.new_station_client)
    monkeypatch.setattr(session_mod, "send_commands", fake_station.send_commands)
    return session_mod.StationSession(host="fake")


def test_connection_status(session):
    async def body():
        await session.wait_for_inference()
        info = session.connection_info()
        assert info["connected"] is True
        assert info["setup_done"] is True
        assert info["has_latest_inference"] is True

    run(body())


def test_get_arm_state_detects_elrobot(session):
    async def body():
        await session.wait_for_inference()
        state = session.get_arm_state()
        assert state["arm_type"] == "elrobot"
        assert len(state["joints"]) == 7
        assert state["gripper"]["motor_id"] == GRIPPER_ID
        assert all(j["torque_enabled"] is False for j in state["joints"])

    run(body())


def test_get_arm_state_home_pose(session):
    async def body():
        await session.wait_for_inference()
        state = session.get_arm_state()
        for joint in state["joints"]:
            motor_id = joint["motor_id"]
            expected = kinematics.radians_to_normalized(kinematics.HOME_JOINT_RAD[motor_id], motor_id)
            assert joint["present_position_normalized"] == pytest.approx(expected, abs=1e-3)

    run(body())


def test_open_close_gripper(session):
    async def body():
        await session.wait_for_inference()
        await session.open_gripper()
        motor = session.get_motor("auto", GRIPPER_ID)
        assert motor["present_position"] == motor["range_min"]

        await session.close_gripper()
        motor = session.get_motor("auto", GRIPPER_ID)
        assert motor["present_position"] == motor["range_max"]

    run(body())


def test_set_gripper_partial(session):
    async def body():
        await session.wait_for_inference()
        result = await session.set_gripper(0.5)
        assert result["gripper_state"] == "partial"
        motor = session.get_motor("auto", GRIPPER_ID)
        assert motor["present_position"] == motor["range_min"] + 0.5 * (motor["range_max"] - motor["range_min"])

    run(body())


def test_move_joint_single(session):
    async def body():
        await session.wait_for_inference()
        await session.move_joint(1, 0.5)
        motor = session.get_motor("auto", 1)
        assert motor["present_position"] == motor["range_min"] + 0.5 * (motor["range_max"] - motor["range_min"])

    run(body())


def test_move_joint_rejects_below_safety_floor(session):
    async def body():
        await session.wait_for_inference()
        before = session.get_motor("auto", 2)["present_position"]

        # joint 2 @ 0.5 (others at home) -- forward_kinematics gives z~=-0.147, below
        # MIN_SAFE_Z_M. This is the exact configuration that motivated extending the
        # workspace floor check to joint-space moves: a real demo motion ("nod") used
        # to jump straight to this value and visibly drove the gripper toward the
        # table, with no protection at all since the floor only covered move_to_xyz.
        with pytest.raises(RuntimeError, match="workspace safety floor"):
            await session.move_joint(2, 0.5)

        after = session.get_motor("auto", 2)["present_position"]
        assert after == before

    run(body())


def test_move_joint_gripper_only_skips_floor_check(session):
    async def body():
        await session.wait_for_inference()
        # Gripper isn't part of the IK chain -- moving it must never be blocked by the
        # workspace floor check, regardless of value.
        await session.set_gripper(0.5)
        assert session.get_motor("auto", fake_station.GRIPPER_MOTOR_ID)["present_position"] > 0

    run(body())


def test_move_arm_pose_multi(session):
    async def body():
        await session.wait_for_inference()
        before_motor3 = session.get_motor("auto", 3)["present_position"]

        # 1=0.3, 5=0.7 chosen via forward_kinematics to stay above MIN_SAFE_Z_M --
        # joint 1 and 2 together (e.g. 0.2/0.8) trips the workspace floor check, which
        # is exactly the new safety behavior, just not what this test is exercising.
        await session.move_arm_pose({1: 0.3, 5: 0.7})

        motor1 = session.get_motor("auto", 1)
        motor5 = session.get_motor("auto", 5)
        motor3 = session.get_motor("auto", 3)
        assert motor1["present_position"] == motor1["range_min"] + 0.3 * (motor1["range_max"] - motor1["range_min"])
        assert motor5["present_position"] == motor5["range_min"] + 0.7 * (motor5["range_max"] - motor5["range_min"])
        assert motor3["present_position"] == before_motor3

    run(body())


def test_enable_disable_arm_torque(session):
    async def body():
        await session.wait_for_inference()
        await session.enable_arm_torque()
        for motor_id in (*ARM_JOINT_IDS, GRIPPER_ID):
            assert session.get_motor("auto", motor_id)["torque_enabled"] is True

        await session.disable_arm_torque()
        for motor_id in (*ARM_JOINT_IDS, GRIPPER_ID):
            assert session.get_motor("auto", motor_id)["torque_enabled"] is False

    run(body())


def test_get_gripper_xyz_at_home(session):
    async def body():
        await session.wait_for_inference()
        result = await session.get_gripper_xyz()
        # Tolerance is looser than kinematics.IK_POSITION_TOLERANCE_M -- seeding round-trips
        # HOME_JOINT_RAD through integer step quantization (normalized_to_steps truncates),
        # which introduces a small reconstruction error this test must absorb.
        for actual, expected in zip(result["xyz"], kinematics.HOME_POSE_XYZ):
            assert actual == pytest.approx(expected, abs=0.01)

    run(body())


def _small_nudge_target() -> list[float]:
    nudged = dict(kinematics.HOME_JOINT_RAD)
    nudged[1] += 0.02
    return list(kinematics.forward_kinematics(nudged))


def _far_target() -> list[float]:
    return list(kinematics.forward_kinematics({i: 0.0 for i in range(1, 8)}))


def test_move_to_xyz_small_delta(session):
    async def body():
        await session.wait_for_inference()
        target = _small_nudge_target()
        await session.move_to_xyz(target)
        result = await session.get_gripper_xyz()
        for actual, expected in zip(result["xyz"], target):
            assert actual == pytest.approx(expected, abs=0.01)

    run(body())


def test_move_to_xyz_seeds_ik_from_current_pose_for_continuity(session):
    async def body():
        await session.wait_for_inference()
        before = (await session.get_gripper_xyz())["joint_rad"]

        # A small +Z-only move from home has no reason to touch joint 7 (wrist roll) --
        # it's redundant for this target. Without seeding solve_ik from the arm's
        # current pose, the optimizer converges to an arbitrary point in that redundant
        # null space each call (always starting from all-zero), so joint 7 could land
        # anywhere; seeded from "where the arm already is," it should stay put instead.
        await session.move_to_xyz([0.0, 0.0, 0.04])

        after = (await session.get_gripper_xyz())["joint_rad"]
        assert after[7] == pytest.approx(before[7], abs=0.01)

    run(body())


def test_move_to_xyz_rejects_large_delta(session):
    async def body():
        await session.wait_for_inference()
        before = await session.get_gripper_xyz()
        with pytest.raises(RuntimeError, match="Refusing move"):
            await session.move_to_xyz(_far_target())
        after = await session.get_gripper_xyz()
        assert after["joint_rad"] == before["joint_rad"]

    run(body())


def test_move_to_xyz_rejects_below_safety_floor(session):
    async def body():
        await session.wait_for_inference()
        before = await session.get_gripper_xyz()
        below_floor = [0.0, 0.0, session_mod.MIN_SAFE_Z_M - 0.05]
        with pytest.raises(RuntimeError, match="workspace safety floor"):
            await session.move_to_xyz(below_floor)
        after = await session.get_gripper_xyz()
        assert after["joint_rad"] == before["joint_rad"]

    run(body())


def test_go_home_from_offset(session):
    async def body():
        await session.wait_for_inference()
        await session.move_to_xyz(_small_nudge_target())

        await session.go_home()
        result = await session.get_gripper_xyz()
        for actual, expected in zip(result["xyz"], kinematics.HOME_POSE_XYZ):
            assert actual == pytest.approx(expected, abs=0.01)

    run(body())


def _second_small_nudge_target() -> list[float]:
    nudged = dict(kinematics.HOME_JOINT_RAD)
    nudged[2] += 0.02
    return list(kinematics.forward_kinematics(nudged))


def test_move_through_waypoints_sequence(session):
    async def body():
        await session.wait_for_inference()
        wp1 = _small_nudge_target()
        wp2 = _second_small_nudge_target()

        result = await session.move_through_waypoints([wp1, wp2])
        assert result["waypoint_count"] == 2
        assert [step["index"] for step in result["steps"]] == [0, 1]

        final = await session.get_gripper_xyz()
        for actual, expected in zip(final["xyz"], wp2):
            assert actual == pytest.approx(expected, abs=0.01)

    run(body())


def test_move_through_waypoints_stops_on_failure(session):
    async def body():
        await session.wait_for_inference()
        wp1 = _small_nudge_target()

        with pytest.raises(RuntimeError, match="Refusing move"):
            await session.move_through_waypoints([wp1, _far_target()])

        after = await session.get_gripper_xyz()
        for actual, expected in zip(after["xyz"], wp1):
            assert actual == pytest.approx(expected, abs=0.01)

    run(body())


def test_pick_at_xyz_sequential_consistency(session):
    async def body():
        await session.wait_for_inference()
        target = _small_nudge_target()
        result = await session.pick_at_xyz(target, approach_height_m=0.01)
        assert [step["step"] for step in result["steps"]] == [
            "open_gripper",
            "approach",
            "descend",
            "close_gripper",
            "retreat",
        ]
        gripper = session.get_motor("auto", GRIPPER_ID)
        assert gripper["present_position"] == gripper["range_max"]

    run(body())


def test_place_at_xyz_sequential_consistency(session):
    async def body():
        await session.wait_for_inference()
        target = _small_nudge_target()
        result = await session.place_at_xyz(target, approach_height_m=0.01)
        assert [step["step"] for step in result["steps"]] == [
            "approach",
            "descend",
            "open_gripper",
            "retreat",
        ]
        gripper = session.get_motor("auto", GRIPPER_ID)
        assert gripper["present_position"] == gripper["range_min"]

    run(body())


def test_move_motor_steps_single_write(session):
    async def body():
        await session.wait_for_inference()
        await session.move_motor_steps(1, 2000)
        assert session.get_motor("auto", 1)["present_position"] == 2000

    run(body())


def test_move_motor_steps_clamps_out_of_range(session):
    async def body():
        await session.wait_for_inference()
        await session.move_motor_steps(1, 9999)
        motor = session.get_motor("auto", 1)
        assert motor["present_position"] == motor["range_max"]

    run(body())


def test_set_torque_specific_motors(session):
    async def body():
        await session.wait_for_inference()
        await session.set_torque([3], True)
        assert session.get_motor("auto", 3)["torque_enabled"] is True
        assert session.get_motor("auto", 4)["torque_enabled"] is False

    run(body())


def test_unknown_motor_id_raises(session):
    async def body():
        await session.wait_for_inference()
        with pytest.raises(RuntimeError, match="Motor 99 not found"):
            session.get_motor("auto", 99)

    run(body())
