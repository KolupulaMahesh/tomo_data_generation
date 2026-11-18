"""What is known about the scene at a frame, kept in three separate kinds (they go to separate files, never mixed):

  truth()       ground truth from the simulation: exact object poses, held or not, part angles. Used for the labels and
                for the generator's own failure checks. NOT a camera input.
  observable()  what the robot itself can know: its joints, its grip IO, whether an object is inside the head camera's
                view (pinhole projection with the live lens) and where it appears in the picture.
  The camera pictures themselves come from the renderer.
"""
from __future__ import annotations

import numpy as np


def project(task, kin, b, P):
    """object position -> (u, v, depth) in the full head-camera picture, or None when behind the camera.
    Head_camera_link is already an optical frame: z forward, x right, y down (checked on boxing_001: the phone lies on
    +z at frames 55 / 100 / 300 and to the right, +x, at 184 when the head turns to the jig)."""
    fx, fy, cx, cy, W, H = task["camera_lens"]
    C = kin.link(task["head"]["camera_link"], b.q, b.base())
    x, y, z = (np.linalg.inv(C) @ np.r_[P[:3, 3], 1.0])[:3]
    if z <= 0.05:
        return None
    return fx * x / z + cx, fy * y / z + cy, z


def seen(task, kin, clear, b, name, half, px=22.0, centre=(0.0, 0.0, 0.0)):
    """how much of the object the head camera really sees now (0..1): points on its top face that are inside the
    picture and not hidden behind any part of either arm / hand (their collision hulls, projected with the live lens:
    a robot point within `px` pixels of an object point and nearer to the camera hides it)."""
    fx, fy, cx, cy, W, H = task["camera_lens"]
    C = np.linalg.inv(kin.link(task["head"]["camera_link"], b.q, b.base()))
    P = b.obj_pose(name)
    hx, hy, hz = half
    cx0, cy0, cz0 = centre
    loc = np.array([[cx0 + x, cy0 + y, cz0 + hz, 1.0] for x in (-hx * .8, 0, hx * .8) for y in (-hy * .8, 0, hy * .8)])
    obj = (C @ P @ loc.T).T[:, :3]
    rob = np.vstack([(C[:3, :3] @ (pts @ T[:3, :3].T + T[:3, 3]).T).T + C[:3, 3]
                     for pts, T in ((clear.pts[n], kin.link(n, b.q, b.base())) for n in clear.left + clear.right)])
    rob = rob[rob[:, 2] > 0.05]
    ru, rv = fx * rob[:, 0] / rob[:, 2] + cx, fy * rob[:, 1] / rob[:, 2] + cy
    ok = 0
    for x, y, z in obj:
        if z <= 0.05:
            continue
        u, v = fx * x / z + cx, fy * y / z + cy
        if not (0 <= u < W and 0 <= v < H):
            continue
        near = (np.abs(ru - u) < px) & (np.abs(rv - v) < px) & (rob[:, 2] < z - 0.01)
        ok += not near.any()
    return ok / len(obj)


def observable(task, kin, b, names):
    out = {"grip_io": int(b.relay)}
    fx, fy, cx, cy, W, H = task["camera_lens"]
    for n in names:
        uv = project(task, kin, b, b.obj_pose(n))
        vis = uv is not None and 0 <= uv[0] < W and 0 <= uv[1] < H
        out[n] = {"visible": bool(vis), "uv": [round(float(uv[0]), 1), round(float(uv[1]), 1)] if vis else None}
    return out


def truth(b, names):
    return {n: {"pos": [round(float(x), 4) for x in b.obj_pose(n)[:3, 3]],
                "held_by": list(b.held[n]) if n in b.held else [],
                "free": n in b.free} for n in names}
