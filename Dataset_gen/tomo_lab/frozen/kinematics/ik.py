"""
tomo_promp/kinematics/ik.py
───────────────────────────
Damped-least-squares inverse kinematics against RobotFK. Pure numpy — no Isaac,
no ikpy, no cuRobo, nothing that cannot be pip-installed on an aarch64 Jetson.

Ported from the laptop repo's isaac_lab_tomo/motion_numpy.py, keeping the exact
same Jacobian, damping and clamping behaviour. Two solvers:

  solve_ik_pose      — 6-DOF: hits a target position AND orientation.
  solve_ik_position  — 3-DOF: position only, orientation left to the solver.

Which to use, and why it matters: position-only IK has zero rotation constraint,
so it can converge to any wrist angle that satisfies the position — that is what
produced the "fingers pointing upward" regression. The reach therefore uses
solve_ik_pose with the rotation taken from the matched demo's own natural wrist
orientation at that frame. Creep uses solve_ik_position because its steps are
sub-centimetre and seeded from the previous solution, so the wrist has no room
to wander.

Both are seeded from the previous waypoint's solution. That is what keeps them
well-conditioned: each call is a small, local correction, not a global search.

Joint limits are enforced INSIDE the iteration (clamped every step), not just at
the end, so the solver cannot walk out of range and then get yanked back.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .fk import RobotFK

# Right-arm joint limits, radians. Sourced from robot/robot_joints.py's
# JOINT_LIMITS (verified 2026-07-20 to match motion_numpy.py — a claimed
# mismatch on Right_arm_joint2 was investigated and RETRACTED, both agree on
# (-2.897247, 0.331613); do not re-raise it).
RIGHT_ARM_LIMITS = {
    "Right_arm_joint1": (-2.949606, 2.949606),
    "Right_arm_joint2": (-2.897247, 0.331613),
    "Right_arm_joint3": (-2.949606, 2.949606),
    "Right_arm_joint4": (-1.553343, 1.553343),
    "Right_arm_joint5": (-2.949606, 2.949606),
    "Right_arm_joint6": (-1.553343, 1.553343),
    "Right_arm_joint7": (-1.553343, 1.553343),
}


def numerical_jacobian_pose(fk: RobotFK, link: str, angles: Dict[str, float],
                            joint_names: List[str], eps: float = 1e-4) -> np.ndarray:
    """(6, len(joint_names)) finite-difference pose Jacobian."""
    base_pos = fk.link_position(link, angles)
    base_rot = fk.link_transform(link, angles)[:3, :3]
    J = np.zeros((6, len(joint_names)))
    for i, jn in enumerate(joint_names):
        perturbed = dict(angles)
        perturbed[jn] = angles.get(jn, 0.0) + eps

        p_pos = fk.link_position(link, perturbed)
        p_rot = fk.link_transform(link, perturbed)[:3, :3]

        dpos = (p_pos - base_pos) / eps

        # rotation difference: p_rot = dR @ base_rot  =>  dR = p_rot @ base_rot.T
        # dR ~= I + skew(omega)*eps  =>  omega*eps = unskew(dR - I)
        dR = p_rot @ base_rot.T
        rx = (dR[2, 1] - dR[1, 2]) / 2.0
        ry = (dR[0, 2] - dR[2, 0]) / 2.0
        rz = (dR[1, 0] - dR[0, 1]) / 2.0
        omega = np.array([rx, ry, rz]) / eps

        J[:3, i] = dpos
        J[3:, i] = omega
    return J


def numerical_jacobian(fk: RobotFK, link: str, angles: Dict[str, float],
                       joint_names: List[str], eps: float = 1e-4) -> np.ndarray:
    """(3, len(joint_names)) finite-difference position Jacobian."""
    base = fk.link_position(link, angles)
    J = np.zeros((3, len(joint_names)))
    for i, jn in enumerate(joint_names):
        perturbed = dict(angles)
        perturbed[jn] = angles.get(jn, 0.0) + eps
        J[:, i] = (fk.link_position(link, perturbed) - base) / eps
    return J


def get_pose_error(cur_mat: np.ndarray, target_mat: np.ndarray) -> np.ndarray:
    """6D error vector [dx, dy, dz, rx, ry, rz]."""
    err = np.zeros(6)
    err[:3] = target_mat[:3, 3] - cur_mat[:3, 3]
    R_err = target_mat[:3, :3] @ cur_mat[:3, :3].T
    err[3] = (R_err[2, 1] - R_err[1, 2]) / 2.0
    err[4] = (R_err[0, 2] - R_err[2, 0]) / 2.0
    err[5] = (R_err[1, 0] - R_err[0, 1]) / 2.0
    return err


def _clamp(jn: str, val: float, limits: Dict[str, tuple] | None = None) -> float:
    lo_hi = (RIGHT_ARM_LIMITS if limits is None else limits).get(jn)
    if lo_hi is None:
        return val
    return max(lo_hi[0], min(lo_hi[1], val))


def solve_ik_pose(fk: RobotFK, link: str, target_mat: np.ndarray,
                  seed_angles: Dict[str, float], joint_names: List[str],
                  iters: int = 150, damping: float = 0.05,
                  limits: Dict[str, tuple] | None = None) -> Dict[str, float]:
    """Damped least-squares IK to a target 6D pose, seeded from seed_angles. limits: {joint: (lo, hi)} kept inside
    every iteration (default RIGHT_ARM_LIMITS: the right arm only, as before; the two-arm high-level control passes
    the robot node's command limits for both arms, brain/hlc_contacts.py)."""
    angles = dict(seed_angles)
    for _ in range(iters):
        cur_mat = fk.link_transform(link, angles)
        err = get_pose_error(cur_mat, target_mat)
        if np.linalg.norm(err) < 1e-4:
            break
        J = numerical_jacobian_pose(fk, link, angles, joint_names)
        JJt = J @ J.T
        dtheta = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6), err)
        for i, jn in enumerate(joint_names):
            angles[jn] = _clamp(jn, angles.get(jn, 0.0) + float(dtheta[i]), limits)
    return angles


def solve_ik_position(fk: RobotFK, link: str, target_pos: np.ndarray,
                      seed_angles: Dict[str, float], joint_names: List[str],
                      iters: int = 100, damping: float = 0.05) -> Dict[str, float]:
    """Damped least-squares IK to a target position, seeded from seed_angles
    (should be the previous waypoint's solution — small steps, well-conditioned)."""
    angles = dict(seed_angles)
    for _ in range(iters):
        cur = fk.link_position(link, angles)
        err = np.asarray(target_pos, dtype=np.float64) - cur
        if np.linalg.norm(err) < 1e-4:
            break
        J = numerical_jacobian(fk, link, angles, joint_names)
        JJt = J @ J.T
        dtheta = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(3), err)
        for i, jn in enumerate(joint_names):
            angles[jn] = _clamp(jn, angles.get(jn, 0.0) + float(dtheta[i]))
    return angles
