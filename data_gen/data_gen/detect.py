"""Failure detection and classification from the scene's actual state (object and hand poses), not from what the
generator meant to do: a check looks at where things ARE in the frame just written.
"""
from __future__ import annotations

import numpy as np

from .kin import angle_deg


class Grasp:
    """One hold from the demo (task_*.yaml grasps), with the demo's own grip offsets."""

    def __init__(self, key, cfg, src, kin, hands):
        self.key, self.c = key, cfg
        self.obj = cfg["object"]
        self.grip, self.release, self.approach = cfg["grip"], cfg["release"], cfg.get("approach", 30)
        self.settle = cfg.get("settle", 15)
        self.available = self.obj in src.objects
        if not self.available:
            self.hands = []
            return
        O = src.objects[self.obj]
        hp = lambda h, i: kin.link(hands[h]["link"], src.q[i], src.robot_world[i])  # noqa: E731
        hand = cfg.get("hand", "auto")
        if hand == "auto":                                     # the hand the object keeps a fixed offset to
            mid = range(self.grip + 3, self.release - 2, 4)
            spread = {h: np.std([(np.linalg.inv(hp(h, i)) @ O[i])[:3, 3] for i in mid], axis=0).sum() for h in hands}
            hand = min(spread, key=spread.get)
        self.hands = ["right", "left"] if hand == "both" else [hand]
        g = (self.grip + self.release) // 2                    # mid-carry: hand and object move together (right after the
        self.rel = {h: np.linalg.inv(hp(h, g)) @ O[g] for h in self.hands}   # grip a strip is still being peeled off)
        self.pre = {h: np.linalg.inv(O[g]) @ hp(h, max(self.grip - self.approach, 0)) for h in self.hands}  # object -> hand before
        self.start_pose = O[self.grip].copy()                  # where it is picked from
        self.target = O[min(self.release + self.settle, len(O) - 1)].copy()   # where it should end up
        self.carry = O[(self.grip + self.release) // 2].copy()
        self.q_open = src.q[max(self.grip - self.approach // 2, 0)].copy()
        self.q_closed = src.q[min(self.grip + 3, len(O) - 1)].copy()
        self.rest_z = float(O[0][2, 3])
        moved = [i for i in range(self.grip, self.release)
                 if np.linalg.norm(O[i][:3, 3] - O[self.grip][:3, 3]) > 0.005]
        self.lift_frame = moved[0] if moved else self.grip + 10      # a strip is peeled before it comes away                        # resting height of its origin on its first surface


def is_held(b, g: Grasp, tol, lifted_from=None):
    """held = the object keeps the demo's offset to the hand (and, if lifted_from is given, it is off that surface)."""
    P = b.obj_pose(g.obj)
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])                         # turned 180 deg about its own z
    for h in g.hands:
        ok = False
        for P2 in ([P, P @ flip] if g.c.get("symmetric_yaw", True) else [P]):
            R = np.linalg.inv(b.hand_pose(h)) @ P2
            ok |= bool(np.linalg.norm(R[:3, 3] - g.rel[h][:3, 3]) <= tol["attached_pos_m"] * 2.5 and
                       angle_deg(R[:3, :3], g.rel[h][:3, :3]) <= tol["attached_rot_deg"] * 2)
        if not ok:
            return False
    if lifted_from is not None and P[2, 3] < lifted_from + 0.01:
        return False
    return True


def placed_error(P, target):
    return float(np.linalg.norm(P[:3, 3] - target[:3, 3])), angle_deg(P[:3, :3], target[:3, :3])


def is_placed(b, g: Grasp, tol, target=None):
    e, r = placed_error(b.obj_pose(g.obj), g.target if target is None else target)
    return e < tol["placed_pos_m"] and r < tol["placed_rot_deg"]


def on_table(task, P):
    (x0, x1), (y0, y1) = task["table_xy"]
    return x0 <= P[0, 3] <= x1 and y0 <= P[1, 3] <= y1 and P[2, 3] > task["table_z"] - 0.02


def part_error_deg(b, part, target_frame):
    """largest angle between the carton part's bones now and in the demo at target_frame, each measured on its own
    hinge (relative to its parent bone: a flap on the lid panel is folded whatever the panel does)."""
    i = b.n - 1
    src = b.out["src"][i]
    worst = 0.0
    for bone in part["bones"]:
        key = (part["armature"], bone)
        par = b.s.bone_parent.get(key)
        cur = b.bone_out[i].get(key, b.s.bones[key][src])
        tgt = b.s.bones[key][target_frame]
        if par:
            pk = (part["armature"], par)
            cur = np.linalg.inv(b.bone_out[i].get(pk, b.s.bones[pk][src])) @ cur
            tgt = np.linalg.inv(b.s.bones[pk][target_frame]) @ tgt
        worst = max(worst, angle_deg(cur[:3, :3], tgt[:3, :3]))
    return worst


def classify_hold(task, b, g: Grasp, tol, expect):
    """expect: 'held' (after a grip) or 'placed' (after a release). Returns the failure type, or None when fine."""
    P = b.obj_pose(g.obj)
    name = g.key
    if expect == "held":
        if is_held(b, g, tol):
            return None
        if not on_table(task, P):
            return f"{name}_off_table"
        if np.linalg.norm(P[:3, 3] - g.start_pose[:3, 3]) < 0.03:
            return f"{name}_not_grasped"
        return f"{name}_dropped"
    e, r = placed_error(P, g.target)
    if e < tol["placed_pos_m"] and r < tol["placed_rot_deg"]:
        return None
    if not on_table(task, P):
        return f"{name}_off_table"
    if e > 0.04:
        return f"{name}_outside_target"
    if r > 12:
        return f"{name}_tilted"
    return f"{name}_partly_misplaced"
