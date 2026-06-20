"""Unit tests for norma_station_mcp.kinematics. Pure math -- no hardware/Station needed.

StationSession integration (move_to_xyz / pick_at_xyz / etc. actually talking to a live
Station) is NOT covered here and is deferred to whenever the physical arm is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from norma_station_mcp import kinematics

HOME_POSE_XYZ = (0.002925, 0.259566, 0.218892)

# A handful of points spanning the workspace, all within the arm's ~0.35m reach.
WORKSPACE_POINTS = [
    (0.15, 0.05, 0.25),
    (0.05, 0.20, 0.10),
    (0.10, -0.10, 0.20),
    (0.0, 0.25, 0.05),
    (0.08, 0.08, 0.30),
    (-0.05, 0.15, 0.15),
]


def test_chain_active_indices():
    chain = kinematics.get_elrobot_chain()
    assert len(chain.links) == 15
    active_indices = [i for i, active in enumerate(chain.active_links_mask) if active]
    assert active_indices == [2, 4, 6, 8, 10, 12, 14]
    expected_names = [f"rev_motor_0{i}" for i in range(1, 8)]
    assert [chain.links[i].name for i in active_indices] == expected_names


def test_home_pose_forward_kinematics():
    xyz = kinematics.forward_kinematics({i: 0.0 for i in range(1, 8)})
    assert np.allclose(xyz, HOME_POSE_XYZ, atol=1e-4)


def test_forward_kinematics_defaults_missing_joints_to_zero():
    assert np.allclose(kinematics.forward_kinematics({}), HOME_POSE_XYZ, atol=1e-4)


@pytest.mark.parametrize("target", WORKSPACE_POINTS)
def test_ik_fk_round_trip(target):
    joint_rad = kinematics.solve_ik(target)
    assert set(joint_rad) == set(range(1, 8))

    solved_xyz = kinematics.forward_kinematics(joint_rad)
    residual = np.linalg.norm(np.array(solved_xyz) - np.array(target))
    assert residual < kinematics.IK_POSITION_TOLERANCE_M

    for motor_id, angle in joint_rad.items():
        lower, upper = kinematics.joint_bounds_rad(motor_id)
        assert lower - 1e-6 <= angle <= upper + 1e-6


@pytest.mark.parametrize("target", [(5.0, 5.0, 5.0), (100.0, 100.0, 100.0)])
def test_unreachable_target_raises(target):
    with pytest.raises(RuntimeError, match="did not converge"):
        kinematics.solve_ik(target)


@pytest.mark.parametrize(
    "bad_target",
    [
        (1.0, 2.0),
        (1.0, 2.0, "x"),
        None,
        (1.0, 2.0, 3.0, 4.0),
    ],
)
def test_malformed_target_raises_value_error(bad_target):
    with pytest.raises(ValueError):
        kinematics.solve_ik(bad_target)


@pytest.mark.parametrize("motor_id", range(1, 8))
@pytest.mark.parametrize("normalized", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_normalized_radians_round_trip(motor_id, normalized):
    angle = kinematics.normalized_to_radians(normalized, motor_id)
    back = kinematics.radians_to_normalized(angle, motor_id)
    assert back == pytest.approx(normalized, abs=1e-9)


@pytest.mark.parametrize("motor_id", range(1, 8))
def test_normalized_bounds_map_to_urdf_bounds(motor_id):
    lower, upper = kinematics.joint_bounds_rad(motor_id)
    assert kinematics.normalized_to_radians(0.0, motor_id) == pytest.approx(lower)
    assert kinematics.normalized_to_radians(1.0, motor_id) == pytest.approx(upper)


def test_radians_to_normalized_out_of_range_raises():
    lower, upper = kinematics.joint_bounds_rad(1)
    with pytest.raises(ValueError):
        kinematics.radians_to_normalized(upper + 1.0, 1)
    with pytest.raises(ValueError):
        kinematics.radians_to_normalized(lower - 1.0, 1)


def test_normalized_to_radians_out_of_range_raises():
    with pytest.raises(ValueError):
        kinematics.normalized_to_radians(1.5, 1)
    with pytest.raises(ValueError):
        kinematics.normalized_to_radians(-0.5, 1)


def test_unknown_motor_id_raises():
    with pytest.raises(ValueError):
        kinematics.joint_bounds_rad(8)
    with pytest.raises(ValueError):
        kinematics.forward_kinematics({8: 0.0})


def test_joint_rad_dict_to_normalized():
    joint_rad = kinematics.solve_ik((0.15, 0.05, 0.25))
    normalized = kinematics.joint_rad_dict_to_normalized(joint_rad)
    assert set(normalized) == set(joint_rad)
    for motor_id, value in normalized.items():
        assert 0.0 <= value <= 1.0


def test_active_links_mask_last_link_stays_active():
    """Regression pin: ikpy 3.4.2 has a check that forces active_links_mask[-1] to False
    via an `is True` identity comparison, which never fires against a numpy.bool_ element.
    This is exactly what keeps joint 7 controllable in our truncated chain. If a future
    ikpy upgrade changes this, this test should fail loudly instead of silently breaking
    joint-7 control.
    """
    chain = kinematics.get_elrobot_chain()
    assert bool(chain.active_links_mask[-1]) is True


def test_normalized_to_steps_compatibility():
    """Sanity check: kinematics output composes with the existing motor_state conversion
    without needing a live Station."""
    from norma_station_mcp.motor_state import normalized_to_steps

    joint_rad = kinematics.solve_ik((0.15, 0.05, 0.25))
    normalized = kinematics.joint_rad_dict_to_normalized(joint_rad)
    for position in normalized.values():
        steps = normalized_to_steps(position, range_min=100, range_max=4000)
        assert 100 <= steps <= 4000
