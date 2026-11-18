"""Gap between the robot's left and right arm (incl. hands and fingers), from the URDF's collision hulls (MMRT's .obj
hulls next to the .stl meshes) placed by FK. The demo itself keeps the arms >= 24 mm apart (boxing, frames 255-265).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class Clearance:
    def __init__(self, urdf, kin, pts_per_link=200):
        self.kin = kin
        base = Path(urdf).parent
        self.pts = {}
        for link in ET.parse(urdf).getroot().iter("link"):
            n = link.get("name")
            if not n.startswith(("Left_", "Right_", "LF_", "RF_")) or n.endswith("arm_link1"):
                continue                                          # the shoulders sit next to the body, never meet
            el = link.find("collision")
            if el is None:
                el = link.find("visual")
            if el is None or el.find("geometry/mesh") is None:
                continue
            obj = (base / el.find("geometry/mesh").get("filename")).with_suffix(".obj")
            if not obj.exists():
                continue
            v = np.array([[float(x) for x in l.split()[1:4]] for l in obj.read_text().splitlines() if l.startswith("v ")])
            o = el.find("origin")
            xyz = np.array([float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            k = min(pts_per_link, len(v))
            self.pts[n] = v[np.random.default_rng(0).choice(len(v), k, replace=False)] + xyz
        self.left = [n for n in self.pts if n.startswith(("Left_", "LF_"))]
        self.right = [n for n in self.pts if n.startswith(("Right_", "RF_"))]

    def _world(self, names, q, W):
        out = []
        for n in names:
            T = self.kin.link(n, q, W)
            out.append(self.pts[n] @ T[:3, :3].T + T[:3, 3])
        return np.vstack(out)

    def gap(self, q, W):
        """smallest distance between any left-arm and any right-arm point (m)."""
        return float(cKDTree(self._world(self.left, q, W)).query(self._world(self.right, q, W))[0].min())
