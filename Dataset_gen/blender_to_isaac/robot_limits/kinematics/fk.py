"""
tomo_promp/kinematics/fk.py
───────────────────────────
Self-contained forward kinematics straight from the robot's own URDF. No
external FK library — none is installed on Thor, and none is needed: this is
the standard homogeneous-transform chain, ~100 lines of numpy.

Ported verbatim (behaviour-identical) from the laptop repo's brain/kinematics.py,
with the repo-relative default path removed so the URDF is always supplied
explicitly from config. Do not "improve" the maths here without re-checking it:
this FK was cross-verified four independent ways (this implementation, ikpy,
DFKI, and cuRobo) and all four agree to 2e-16 m. It is the one part of the stack
that has never been the problem.

For each joint from base to the target link: apply the joint's fixed origin
transform, then its motion (a rotation about its axis for revolute/continuous
joints, a translation for prismatic, nothing for fixed).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


# ── transform helpers ─────────────────────────────────────────────────────────

def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis RPY -> 3x3 rotation (R = Rz(y) @ Ry(p) @ Rx(r))."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle_to_R(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis by theta radians."""
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-9:
        return np.eye(3)
    a = a / n
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _T(xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = xyz
    return M


# ── URDF model ────────────────────────────────────────────────────────────────

@dataclass
class _Joint:
    name:   str
    jtype:  str
    parent: str
    child:  str
    xyz:    np.ndarray
    rpy:    np.ndarray
    axis:   np.ndarray


class RobotFK:
    """Forward kinematics for full_robot.urdf."""

    def __init__(self, urdf_path: str | Path):
        urdf_path = Path(urdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        root = ET.parse(str(urdf_path)).getroot()
        self._by_child: Dict[str, _Joint] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = (np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
                   if o is not None else np.zeros(3))
            rpy = (np.array([float(v) for v in o.get("rpy", "0 0 0").split()])
                   if o is not None else np.zeros(3))
            ax = j.find("axis")
            axis = (np.array([float(v) for v in ax.get("xyz", "1 0 0").split()])
                    if ax is not None else np.array([1.0, 0.0, 0.0]))
            joint = _Joint(
                name=j.get("name"), jtype=j.get("type", "fixed"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=xyz, rpy=rpy, axis=axis,
            )
            self._by_child[joint.child] = joint

    def has_link(self, link: str) -> bool:
        """True if `link` is reachable in this URDF (it is either a joint child
        or the root). Used to fail loudly at startup on a config typo instead of
        silently returning the identity transform."""
        if link in self._by_child:
            return True
        parents = {j.parent for j in self._by_child.values()}
        return link in parents

    # ── chain from base up to a link ──────────────────────────────────────────

    def _chain(self, link: str) -> List[_Joint]:
        chain: List[_Joint] = []
        seen = set()
        while link in self._by_child and link not in seen:
            seen.add(link)
            j = self._by_child[link]
            chain.append(j)
            link = j.parent
        chain.reverse()          # base -> ... -> target
        return chain

    # ── FK: transform of `link` given joint angles (name -> radians) ──────────

    def link_transform(self, link: str, angles: Dict[str, float]) -> np.ndarray:
        T = np.eye(4)
        for j in self._chain(link):
            T = T @ _T(j.xyz, _rpy_to_R(*j.rpy))
            if j.jtype in ("revolute", "continuous"):
                theta = float(angles.get(j.name, 0.0))
                T = T @ _T(np.zeros(3), _axis_angle_to_R(j.axis, theta))
            elif j.jtype == "prismatic":
                d = float(angles.get(j.name, 0.0))
                T = T @ _T(j.axis * d / (np.linalg.norm(j.axis) + 1e-9), np.eye(3))
            # fixed -> nothing
        return T

    def link_position(self, link: str, angles: Dict[str, float]) -> np.ndarray:
        return self.link_transform(link, angles)[:3, 3]
