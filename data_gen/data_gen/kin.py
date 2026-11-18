"""Forward and inverse kinematics of the robot from its URDF, inside the URDF joint limits.

FK: blender_to_isaac/robot_limits/kinematics/fk.py (the brain's own, cross-checked 4 ways), used as is.
IK: damped least squares on one arm (7 joints) to a 6-D hand pose; the joints are clamped to the URDF limits on every
iteration, so a solution never leaves them. The other joints stay as given.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Dataset_gen/blender_to_isaac/robot_limits"))
from kinematics.fk import RobotFK  # noqa: E402


def urdf_limits(urdf):
    out = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        lim = j.find("limit")
        if j.get("type") in ("revolute", "prismatic") and lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def rot_err(Ra, Rb):
    """rotation vector taking Ra to Rb (world frame)."""
    R = Rb @ Ra.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(a)
    if s < 1e-6:                                         # ~180 deg
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        return V[:, -1] * a
    return v / (2 * s) * a


def angle_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(rot_err(Ra, Rb))))


def slerp(Ra, Rb, t):
    v = rot_err(Ra, Rb) * t
    a = np.linalg.norm(v)
    if a < 1e-12:
        return Ra.copy()
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K) @ Ra


def interp_pose(A, B, t):
    M = np.eye(4)
    M[:3, :3] = slerp(A[:3, :3], B[:3, :3], t)
    M[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]
    return M


LIVE_URDF = Path(__file__).resolve().parents[2] / "Dataset_gen/blender_to_isaac/robot_limits/source_gidora/robot_description.urdf"


class Kin:
    """Joint limits (ROBOT LIMITS ARE NEVER CROSSED):
      lim       hard limits = for each joint the stricter of the given URDF and the live robot's own URDF
      lim_safe  the limits every joint value the generator makes (IK, blends, finger moves) is clamped to: the hard
                limits themselves (margin_deg = 0, user's rule 2026-11-17: inside the robot's limits, no extra margin). The demo's own frames are as recorded
                (inside the hard limits; validate.py checks every frame of every episode)."""

    def __init__(self, urdf, joints, margin_deg=0.0, extra_urdfs=(LIVE_URDF,)):
        self.fk = RobotFK(urdf)
        self.joints = list(joints)
        self.lim = urdf_limits(urdf)
        for u in extra_urdfs:
            if Path(u).is_file():
                for n, (lo, hi) in urdf_limits(u).items():
                    if n in self.lim:
                        self.lim[n] = (max(self.lim[n][0], lo), min(self.lim[n][1], hi))
        m = np.radians(margin_deg)
        self.margin = m
        self.lim_safe = {n: (lo + m, hi - m) if hi - lo > 2 * m else ((lo + hi) / 2, (lo + hi) / 2)
                         for n, (lo, hi) in self.lim.items()}

    def angles(self, q):
        return dict(zip(self.joints, map(float, q)))

    def link(self, link, q, base=None):
        T = self.fk.link_transform(link, self.angles(q))
        return T if base is None else base @ T

    def clamp(self, q, safe=True):
        """into the limits: the safe ones (2 deg in) for everything the generator makes."""
        lim = self.lim_safe if safe else self.lim
        q = np.array(q, float)
        for i, n in enumerate(self.joints):
            if n in lim:
                q[i] = np.clip(q[i], *lim[n])
        return q

    def within_limits(self, q, tol=1e-6, safe=False):
        """(ok, [(joint, min, max, lo, hi)]) for one frame or (T, J); safe=True checks the 2-deg-in limits."""
        lim = self.lim_safe if safe else self.lim
        q = np.atleast_2d(q)
        bad = []
        for i, n in enumerate(self.joints):
            if n in lim:
                lo, hi = lim[n]
                if (q[:, i] < lo - tol).any() or (q[:, i] > hi + tol).any():
                    bad.append((n, float(q[:, i].min()), float(q[:, i].max()), lo, hi))
        return not bad, bad

    def ik(self, link, arm, target, q0, base=None, iters=150, damp=0.03, tol_p=1e-3, tol_r=np.radians(1.0)):
        """q (J,) with the arm joints moved so `link` reaches target (world 4x4). Returns (q, pos_err_m, rot_err_deg)."""
        idx = [self.joints.index(n) for n in arm]
        tgt = target if base is None else np.linalg.inv(base) @ target
        q = self.clamp(q0)
        eps = 1e-5
        for _ in range(iters):
            cur = self.link(link, q)
            e = np.r_[tgt[:3, 3] - cur[:3, 3], rot_err(cur[:3, :3], tgt[:3, :3])]
            if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_r:
                break
            J = np.zeros((6, len(idx)))
            for c, i in enumerate(idx):
                qq = q.copy()
                qq[i] += eps
                Tn = self.link(link, qq)
                J[:3, c] = (Tn[:3, 3] - cur[:3, 3]) / eps
                J[3:, c] = rot_err(cur[:3, :3], Tn[:3, :3]) / eps
            W = np.diag([1, 1, 1, 0.3, 0.3, 0.3])                  # position first, orientation a bit softer
            JW = W @ J
            dq = JW.T @ np.linalg.solve(JW @ JW.T + damp ** 2 * np.eye(6), W @ e)
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q[idx] += dq
            q = self.clamp(q)
        cur = self.link(link, q)
        return q, float(np.linalg.norm(tgt[:3, 3] - cur[:3, 3])), angle_deg(cur[:3, :3], tgt[:3, :3])
