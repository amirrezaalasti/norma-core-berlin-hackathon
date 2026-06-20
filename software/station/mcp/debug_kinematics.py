"""Offline kinematics diagnostic for the "commanded up, arm went sideways" bug.

Pure math only -- no Station, no `--tcp`, no hardware needed. This is the opposite
of `test_small_move.py` (which drives the real arm); this script only ever calls
into `norma_station_mcp.kinematics`.

Run:

    uv sync --project software/station/mcp
    uv run --project software/station/mcp python software/station/mcp/debug_kinematics.py

What this can prove: whether the IK/URDF math itself produces a geometrically sane
vertical lift for a +Z target, or whether the math/URDF already swings the base
sideways in pure simulation -- independent of any real hardware.

What this cannot prove: whether a real motor actually rotates in the URDF's assumed
positive-axis direction for a given normalized command. That's the
"hardware-unverifiable assumption" flagged in kinematics.py's module docstring, and
it can only be checked by bench-testing the physical arm one joint at a time (see
the recipe printed at the end of this script).
"""

from __future__ import annotations

import math

from norma_station_mcp import kinematics

# Mirrors test_small_move.py's DELTA_Z -- keep in sync with that file.
DELTA_Z = 0.03

JOINT_IDS = list(range(1, 8))

# Dominant URDF axis per motor, for cross-reference against the numeric sensitivity
# table below. Source: hardware/elrobot/simulation/elrobot_follower.urdf <axis> tags.
URDF_AXIS_LABEL = {
    1: "+Z",
    2: "-X",
    3: "-X",
    4: "-X",
    5: "-Y",
    6: "-X",
    7: "-Y",
}


def _home_pose() -> dict[int, float]:
    return {i: 0.0 for i in JOINT_IDS}


def reproduce_bug_move() -> tuple[dict[int, float], dict[int, float]]:
    """Reproduce test_small_move.py's "lift gripper by +0.03m Z" purely in simulation.

    Returns (before_joint_rad, after_joint_rad).
    """
    before = _home_pose()
    before_xyz = kinematics.forward_kinematics(before)
    target = (before_xyz[0], before_xyz[1], before_xyz[2] + DELTA_Z)

    after = kinematics.solve_ik(target, initial_joint_rad=before)
    after_xyz = kinematics.forward_kinematics(after)
    residual = math.dist(after_xyz, target)

    print("=" * 70)
    print("REPRODUCING test_small_move.py: home pose, lift +%.3fm in Z" % DELTA_Z)
    print("=" * 70)
    print(f"before xyz: {tuple(round(v, 4) for v in before_xyz)}")
    print(f"target xyz: {tuple(round(v, 4) for v in target)}")
    print(f"after  xyz: {tuple(round(v, 4) for v in after_xyz)}  (residual {residual:.5f} m)")
    print()
    print(f"{'motor':>5}  {'before(rad)':>11}  {'after(rad)':>10}  {'delta(deg)':>10}  {'bound(rad)':>20}  near-bound")
    for motor_id in JOINT_IDS:
        lower, upper = kinematics.joint_bounds_rad(motor_id)
        b = before[motor_id]
        a = after[motor_id]
        delta_deg = math.degrees(a - b)
        span = upper - lower
        near_bound = (a - lower) / span < 0.05 or (upper - a) / span < 0.05
        print(
            f"{motor_id:>5}  {b:>11.4f}  {a:>10.4f}  {delta_deg:>10.2f}  "
            f"[{lower:>8.4f},{upper:>8.4f}]  {'YES' if near_bound else ''}"
        )
    print()
    return before, after


def joint_direction_sensitivity_table(
    base_joint_rad: dict[int, float], delta_rad: float = 1e-3, *, label: str = ""
) -> dict[int, tuple[float, float, float]]:
    """Per-joint d(XYZ)/d(angle) unit direction at base_joint_rad, via central
    differences on forward_kinematics (ikpy exposes no analytic Jacobian).

    Clamps each probe to the joint's URDF bounds, falling back to a one-sided
    difference near a bound (relevant for motors 5/6/7, whose ranges aren't
    symmetric around 0).
    """
    directions: dict[int, tuple[float, float, float]] = {}
    print(f"--- joint direction sensitivity{f' ({label})' if label else ''} ---")
    print(f"{'motor':>5}  {'unit direction (dx,dy,dz)':>32}  {'dominant':>10}  {'urdf axis':>9}")
    for motor_id in JOINT_IDS:
        lower, upper = kinematics.joint_bounds_rad(motor_id)
        angle = base_joint_rad.get(motor_id, 0.0)

        plus_angle = min(angle + delta_rad, upper)
        minus_angle = max(angle - delta_rad, lower)
        step = plus_angle - minus_angle
        if step <= 0:
            directions[motor_id] = (0.0, 0.0, 0.0)
            print(f"{motor_id:>5}  {'(joint pinned at bound)':>32}")
            continue

        plus_pose = dict(base_joint_rad)
        plus_pose[motor_id] = plus_angle
        minus_pose = dict(base_joint_rad)
        minus_pose[motor_id] = minus_angle

        plus_xyz = kinematics.forward_kinematics(plus_pose)
        minus_xyz = kinematics.forward_kinematics(minus_pose)
        velocity = tuple((p - m) / step for p, m in zip(plus_xyz, minus_xyz))
        norm = math.sqrt(sum(v * v for v in velocity))
        unit = tuple(v / norm for v in velocity) if norm > 1e-12 else (0.0, 0.0, 0.0)
        directions[motor_id] = unit

        axis_names = ("X", "Y", "Z")
        dominant_idx = max(range(3), key=lambda i: abs(unit[i]))
        dominant = (
            f"{'+' if unit[dominant_idx] >= 0 else '-'}{axis_names[dominant_idx]}"
            if abs(unit[dominant_idx]) >= 0.8
            else "mixed"
        )
        print(
            f"{motor_id:>5}  ({unit[0]:>7.3f},{unit[1]:>7.3f},{unit[2]:>7.3f})  "
            f"{dominant:>10}  {URDF_AXIS_LABEL[motor_id]:>9}"
        )
    print()
    return directions


def plot_poses(poses: list[tuple[str, dict[int, float]]]) -> None:
    try:
        from ikpy.utils import plot as ikpy_plot
    except ImportError:
        print(
            "matplotlib/ikpy plotting unavailable -- run `uv sync` in "
            "software/station/mcp to install the dev dependency group. "
            "Skipping the 3D plot; printed tables above are unaffected."
        )
        return

    chain = kinematics.get_elrobot_chain()
    _, ax = ikpy_plot.init_3d_figure()
    for label, joint_rad in poses:
        vector = kinematics._full_vector_from_joint_rad(joint_rad)
        ikpy_plot.plot_chain(chain, vector, ax, name=label)
    # init_3d_figure() hardcodes a +/-1.0m frame, far larger than this ~0.35m-reach
    # arm -- without rescaling, the whole chain renders as a single tiny blob.
    reach = 0.4
    ax.set_xlim3d([-reach, reach])
    ax.set_ylim3d([-reach, reach])
    ax.set_zlim3d([-reach, reach])
    ax.legend()
    ikpy_plot.show_figure()


CAVEAT = """
======================================================================
WHAT THIS DOES AND DOES NOT PROVE
======================================================================
CAN prove (pure offline math): whether solve_ik's solution for "home + {dz}m Z"
is itself a geometrically sane vertical lift, or whether the math/URDF already
swings the base sideways -- i.e. whether the bug reproduces in simulation alone.

CANNOT prove (needs real hardware): whether a real motor actually rotates in the
URDF's assumed positive-axis direction for a given normalized command. A mismatch
there would make this simulation look perfectly clean while the real arm still
moves sideways -- see the "Hardware-unverifiable assumption" in kinematics.py's
module docstring.

INTERPRETATION: small deltas concentrated in motors 5/6 above, no large motor-1
swing -> the math looks sane, so the real-world bug is more likely a hardware
calibration-direction mismatch (next step: hardware bench test below). A large,
unexpected motor-1 delta for a pure +Z target instead -> the IK/URDF math itself
is suspect.

NEXT STEPS ONCE HARDWARE IS AVAILABLE:
  1. Power the arm, set every joint to normalized 0.5, let it settle.
  2. Re-run joint_direction_sensitivity_table() at that pose (convert 0.5 per
     joint via kinematics.normalized_to_radians) for a baseline-matched table.
  3. One joint at a time: nudge 0.5 -> 0.6 via move_joint, observe the real
     direction the gripper moves, then return it to 0.5 before the next joint.
  4. Compare observed vs. predicted direction per joint -- whichever disagrees
     (sign and/or dominant axis) is the prime suspect for a follow-up fix.
======================================================================
""".format(dz=DELTA_Z)


def main() -> None:
    before, after = reproduce_bug_move()
    joint_direction_sensitivity_table(before, label="home pose")
    joint_direction_sensitivity_table(after, label="post-lift pose")
    print(CAVEAT)
    plot_poses([("home", before), (f"after +{DELTA_Z}m Z", after)])


if __name__ == "__main__":
    main()
