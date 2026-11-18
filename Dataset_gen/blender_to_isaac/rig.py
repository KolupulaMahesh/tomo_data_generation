"""The packing scene's rig evaluated the way Blender 5.2 does it (object_update / armature_update / constraint.cc), from
rig.json (blender_scripts/export_rig.py -> blender_export/). Pure numpy, so it runs in Isaac Sim's Python.

Per frame: F-curves -> transform channels and constraint influences; objects: local (loc / rot / scale + deltas) under
their parent (object or bone tail) and parent-inverse; armatures: pose bones from rest offsets + channels; then each
owner's constraint stack in order (Child Of, Damped Track, Transform, Limit Rotation, Stretch To), blended by influence.
IK is part 3: until then, the pose of the armatures that have IK is taken from `inject` (e.g. Blender's own trace).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

import ik as IK
from blender_math import (ROT_ORDERS, axis_angle_to_mat3, eul_to_mat3, interp_m4, mat3_normalized_to_eul, mat_to_eul,
                          mat4_to_loc_rot_size, mat4_to_size, normalize_cols, quat_normalize, quat_to_mat3,
                          rotation_between_vecs, trs)
from fcurves import FCurve

TRACK = {"TRACK_X": (0, 1), "TRACK_Y": (1, 1), "TRACK_Z": (2, 1),
         "TRACK_NEGATIVE_X": (0, -1), "TRACK_NEGATIVE_Y": (1, -1), "TRACK_NEGATIVE_Z": (2, -1)}
TRACK_ORDER = ["TRACK_X", "TRACK_Y", "TRACK_Z", "TRACK_NEGATIVE_X", "TRACK_NEGATIVE_Y", "TRACK_NEGATIVE_Z"]
BONE_PATH = re.compile(r'pose\.bones\["(.+?)"\]\.(.+)')
CON_PATH = re.compile(r'constraints\["(.+?)"\]\.influence')


def _A(m):
    return np.array(m, float)


class Rig:
    def __init__(self, rig_json: Path):
        d = json.loads(Path(rig_json).read_text())
        self.doc, self.O = d, d["objects"]
        self.curves = {}                                  # owner -> [(FCurve, data_path, index)]
        for name, o in self.O.items():
            if o["animation"]:
                self.curves[name] = [FCurve(f) for f in o["animation"]["fcurves"]]
        self.bone = {}                                    # (arm, bone) -> rest data
        for name, o in self.O.items():
            if o["type"] != "ARMATURE":
                continue
            for b, bd in o["bones"].items():
                ml = _A(bd["matrix_local"])
                par = bd["parent"]
                offs = np.linalg.inv(_A(o["bones"][par]["matrix_local"])) @ ml if par else ml
                self.bone[(name, b)] = {"parent": par, "ml": ml, "offs": offs, "connect": bd["use_connect"],
                                        "length": float(np.linalg.norm(np.subtract(bd["tail_local"], bd["head_local"])))}
        self.pinv = {k: _A(o["matrix_parent_inverse"]) for k, o in self.O.items()}
        # IK trees (BIK_init_tree / initialize_posetree): the chain from the constrained bone (its tail is the tip) up
        # chain_count bones, root first. One constraint per tree here (the rig's chains have different roots).
        self.ik_tree = {}                                 # (arm, bone) -> tree
        for name, o in self.O.items():
            if o["type"] != "ARMATURE" or o["ik_solver"] != "LEGACY":
                if o["type"] == "ARMATURE" and o["in_scene"] and any(
                        c["type"] == "IK" and not c.get("mute") for pb in o["pose_bones"].values()
                        for c in pb["constraints"]):
                    raise NotImplementedError(f"{name}: IK solver {o['ik_solver']} (only Blender's standard IK is rebuilt)")
                continue
            for b, pb in o["pose_bones"].items():
                for c in pb["constraints"]:
                    if c["type"] != "IK" or c.get("mute") or c.get("enabled") is False:
                        continue
                    tip = b if c["use_tail"] else self.bone[(name, b)]["parent"]
                    chain, cur = [], tip
                    while cur is not None:
                        chain.append(cur)
                        if len(chain) == c["chain_count"] or len(chain) > 255:
                            break
                        cur = self.bone[(name, cur)]["parent"]
                    tree = {"arm": name, "chain": chain[::-1], "con": c, "owner": b}
                    for x in tree["chain"]:
                        if (name, x) in self.ik_tree:
                            raise NotImplementedError(f"IK chains sharing bones ({name} {x})")
                        self.ik_tree[(name, x)] = tree

    # ------------------------------------------------------------------ per frame
    def evaluate(self, frame: float, inject: dict | None = None, overrides: dict | None = None):
        """Start a frame. inject: {(armature, bone): 4x4 pose matrix} used as is (part 2: the IK armatures);
        overrides: {object: {"location": [...], ...}} extra channel values (the generator's moves, later)."""
        self.frame, self.inject, self.overrides = float(frame), inject or {}, overrides or {}
        self._W, self._P, self._props, self._ik_started, self.ik_iterations = {}, {}, {}, set(), {}
        return self

    def props(self, owner):
        """Channel values of an object and its pose bones at this frame: frame-0 values, then the F-curves."""
        if owner in self._props:
            return self._props[owner]
        o = self.O[owner]
        p = {"obj": {k: list(o[k]) for k in ("location", "rotation_euler", "rotation_quaternion", "scale",
                                              "delta_location", "delta_rotation_euler", "delta_rotation_quaternion",
                                              "delta_scale")},
             "bones": {}, "influence": {c["name"]: c["influence"] for c in o["constraints"]}}
        if o["type"] == "ARMATURE":
            for b, pb in o["pose_bones"].items():
                p["bones"][b] = {k: list(pb[k]) for k in ("location", "rotation_quaternion", "rotation_euler", "scale")}
                p["bones"][b]["influence"] = {c["name"]: c["influence"] for c in pb["constraints"]}
        for fc in self.curves.get(owner, []):
            v = fc(self.frame)
            m = BONE_PATH.match(fc.data_path)
            if m:
                b, rest = m.groups()
                cm = CON_PATH.match(rest)
                if cm:
                    p["bones"][b]["influence"][cm.group(1)] = v
                else:
                    p["bones"][b][rest][fc.index] = v
                continue
            cm = CON_PATH.match(fc.data_path)
            if cm:
                p["influence"][cm.group(1)] = v
            else:
                p["obj"][fc.data_path][fc.index] = v
        for k, v in self.overrides.get(owner, {}).items():
            p["obj"][k] = list(v)
        self._props[owner] = p
        return p

    @staticmethod
    def _rot(mode, eul, quat):
        if mode in ROT_ORDERS:
            return eul_to_mat3(eul, mode)
        if mode == "QUATERNION":
            return quat_to_mat3(quat_normalize(quat))
        raise NotImplementedError(f"rotation mode {mode}")

    def basis(self, name):
        """BKE_object_to_mat4: rot = delta_rot @ rot, scale = scale * delta_scale, loc = loc + delta_loc."""
        o, p = self.O[name], self.props(name)["obj"]
        mode = o["rotation_mode"]
        R = self._rot(mode, p["delta_rotation_euler"], p["delta_rotation_quaternion"]) @ \
            self._rot(mode, p["rotation_euler"], p["rotation_quaternion"])
        return trs(np.add(p["location"], p["delta_location"]), R, np.multiply(p["scale"], p["delta_scale"]))

    def chan(self, arm, b):
        """BKE_pchan_to_mat4 (a connected bone ignores its location)."""
        p = self.props(arm)["bones"][b]
        M = trs([0, 0, 0], self._rot(self.O[arm]["pose_bones"][b]["rotation_mode"], p["rotation_euler"],
                                      p["rotation_quaternion"]), p["scale"])
        if not self.bone[(arm, b)]["connect"]:
            M[:3, 3] = p["location"]
        return M

    # ------------------------------------------------------------------ objects
    def world(self, name):
        if name in self._W:
            return self._W[name]
        o = self.O[name]
        M = self.basis(name)
        if o["parent"] and (o["parent_type"] not in ("OBJECT", "BONE", "ARMATURE")
                            or self.O[o["parent"]]["type"] == "CURVE"):            # (a curve parent can be a path)
            raise NotImplementedError(f"{name}: parent type {o['parent_type']} on a {self.O[o['parent']]['type']}")
        if o["parent"]:
            P = self.world(o["parent"])
            if o["parent_type"] == "BONE" and o["parent_bone"]:
                B = self.pose(o["parent"], o["parent_bone"]).copy()
                B[:3, 3] += B[:3, 1] * self.bone[(o["parent"], o["parent_bone"])]["length"]   # child sits at the tail
                P = P @ B
            M = P @ self.pinv[name] @ M
        self._W[name] = self.solve(o["constraints"], M, self.props(name)["influence"], ("obj", name))
        return self._W[name]

    # ------------------------------------------------------------------ armatures
    def parent_transform(self, arm, b):
        """Pose-space transform the bone's own channels sit in (parent pose @ rest offset, or rest for a root bone)."""
        bd = self.bone[(arm, b)]
        return self.pose(arm, bd["parent"]) @ bd["offs"] if bd["parent"] else bd["ml"]

    def pose(self, arm, b):
        key = (arm, b)
        if key in self._P:
            return self._P[key]
        if key in self.inject:
            self._P[key] = np.asarray(self.inject[key], float)
            return self._P[key]
        tree = self.ik_tree.get(key)
        if tree is not None and id(tree) not in self._ik_started:
            self._solve_ik(tree)
            return self._P[key]
        return self._pose_fk(arm, b)

    def _pose_fk(self, arm, b):
        """Rest offset + channels, then the bone's constraints except IK (BKE_pose_where_is_bone)."""
        key = (arm, b)
        M = self.parent_transform(arm, b) @ self.chan(arm, b)
        cons = [c for c in self.O[arm]["pose_bones"][b]["constraints"] if c["type"] != "IK"]
        if cons:
            Wa = self.world(arm)
            M = np.linalg.inv(Wa) @ self.solve(cons, Wa @ M, self.props(arm)["bones"][b]["influence"], ("bone", arm, b))
        self._P[key] = M
        return M

    # ------------------------------------------------------------------ IK (iksolver_plugin.cc: execute_posetree)
    def _solve_ik(self, tree):
        """The chain posed without IK first (keys + its other constraints), the solver on the change of each bone's
        basis, then pose = parent pose @ (pre-IK delta from the parent) @ basis change, root first."""
        self._ik_started.add(id(tree))
        arm, chain, c = tree["arm"], tree["chain"], tree["con"]
        pre = {}
        for b in chain:                                       # pre-IK pose, root first (parents from self._P)
            pre[b] = self._pose_fk(arm, b)
        segs = []
        for a, b in enumerate(chain):
            bd, pb = self.bone[(arm, b)], self.O[arm]["pose_bones"][b]
            par = bd["parent"]
            Rb = pre[b][:3, :3]
            Rp = self.pose(arm, par)[:3, :3] if par else np.eye(3)
            if a > 0 and par:
                Pp = self.pose(arm, par)
                start = pre[b][:3, 3] - (Pp[:3, 3] + Pp[:3, 1] * self.bone[(arm, par)]["length"])
            else:
                start = np.zeros(3)
            length = bd["length"] * float(np.linalg.norm(Rb[:, 1]))
            Rb, Rp = normalize_cols(Rb), normalize_cols(Rp)
            rest = bd["offs"][:3, :3] if par else bd["ml"][:3, :3]          # bone->bone_mat
            iRp = np.linalg.inv(Rp)
            basis = rest.T @ (iRp @ Rb)
            start = normalize_cols(iRp) @ start
            free = [i for i, a_ in enumerate("xyz") if not pb[f"lock_ik_{a_}"] and
                    (not pb[f"use_ik_limit_{a_}"] or pb[f"ik_min_{a_}"] < 0.0 or pb[f"ik_max_{a_}"] > 0.0)]
            if len(free) > 1:
                raise NotImplementedError(f"IK bone {b} with {len(free)} rotation axes (the robot's are one-axis)")
            s = IK.Segment(free[0] if free else None, start, rest, basis, length)
            for i, a_ in enumerate("xyz"):
                if pb[f"use_ik_limit_{a_}"]:
                    s.set_limit(i, pb[f"ik_min_{a_}"], pb[f"ik_max_{a_}"])
                s.set_stiffness(i, pb[f"ik_stiffness_{a_}"])
            segs.append(s)
        par0 = self.bone[(arm, chain[0])]["parent"]
        root = np.eye(4)
        if par0:
            root[:3, :3] = normalize_cols(self.pose(arm, par0)[:3, :3])
        root[:3, 3] = pre[chain[0]][:3, 3]
        goal = np.linalg.inv(self.world(arm) @ root) @ self._target_world(c)
        iters = IK.solve(segs, goal[:3, 3] if c["use_location"] else None, c["weight"],
                         normalize_cols(goal[:3, :3]) if c["use_rotation"] else None, c["orient_weight"],
                         int(c["iterations"]))
        self.ik_iterations[tree["owner"]] = iters
        enf = float(self.props(arm)["bones"][tree["owner"]]["influence"].get(c["name"], c["influence"]))
        delta = {}
        for b in chain:                                       # make_dmats (parents still pre-IK)
            par = self.bone[(arm, b)]["parent"]
            delta[b] = np.linalg.inv(pre[par] if par in pre else self.pose(arm, par)) @ pre[b] if par else pre[b]
        for b, s in zip(chain, segs):                         # where_is_ik_bone, root first
            ch = s.basis_change()
            if enf != 1.0:
                ch = interp_m4(np.eye(4), np.block([[ch, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]),
                               enf)[:3, :3]
            ikm = np.eye(4)
            ikm[:3, :3] = ch
            par = self.bone[(arm, b)]["parent"]
            self._P[(arm, b)] = (self.pose(arm, par) @ delta[b] @ ikm) if par else delta[b] @ ikm

    def _target_world(self, c):
        """The IK target's world matrix (a bone target: armature world @ its pose)."""
        t, sub = c["target"], c.get("subtarget") or ""
        W = self.world(t)
        return W @ self.pose(t, sub) if self.O[t]["type"] == "ARMATURE" and sub else W

    # ------------------------------------------------------------------ spaces
    def to_space(self, owner, M, space):
        """World matrix -> the constraint's space (BKE_constraint_mat_convertspace)."""
        if space == "WORLD":
            return M
        if owner[0] == "obj":
            o = self.O[owner[1]]
            if space == "LOCAL" and o["parent"]:
                return np.linalg.inv(self.world(o["parent"]) @ self.pinv[owner[1]]) @ M
            if space == "LOCAL":
                return M
            raise NotImplementedError(f"object space {space}")
        _, arm, b = owner
        Mp = np.linalg.inv(self.world(arm)) @ M
        if space == "POSE":
            return Mp
        if space == "LOCAL":
            return np.linalg.inv(self.parent_transform(arm, b)) @ Mp
        if space == "LOCAL_WITH_PARENT":
            return np.linalg.inv(self.bone[(arm, b)]["ml"]) @ Mp
        raise NotImplementedError(f"bone space {space}")

    def from_space(self, owner, M, space):
        if space == "WORLD":
            return M
        if owner[0] == "obj":
            o = self.O[owner[1]]
            if space == "LOCAL" and o["parent"]:
                return self.world(o["parent"]) @ self.pinv[owner[1]] @ M
            return M
        _, arm, b = owner
        if space == "POSE":
            Mp = M
        elif space == "LOCAL":
            Mp = self.parent_transform(arm, b) @ M
        elif space == "LOCAL_WITH_PARENT":
            Mp = self.bone[(arm, b)]["ml"] @ M
        else:
            raise NotImplementedError(f"bone space {space}")
        return self.world(arm) @ Mp

    def target(self, c):
        """The constraint target's matrix in its target space (constraint_target_to_mat4 + convertspace)."""
        t, sub = c.get("target"), c.get("subtarget") or ""
        if not t:
            return None
        W = self.world(t)
        if self.O[t]["type"] == "ARMATURE" and sub:
            P = self.pose(t, sub).copy()
            ht = float(c.get("head_tail", 0.0) or 0.0)
            if ht > 1e-6:
                P[:3, 3] += P[:3, 1] * self.bone[(t, sub)]["length"] * ht
            M, owner = W @ P, ("bone", t, sub)
        else:
            M, owner = W, ("obj", t)
        return self.to_space(owner, M, c.get("target_space", "WORLD"))

    def rot_order(self, owner):
        mode = (self.O[owner[1]]["rotation_mode"] if owner[0] == "obj"
                else self.O[owner[1]]["pose_bones"][owner[2]]["rotation_mode"])
        return mode if mode in ROT_ORDERS else "XYZ"

    # ------------------------------------------------------------------ the constraint stack
    def solve(self, constraints, M, influence, owner):
        old = M
        for c in constraints:
            if c["type"] == "IK" or c.get("mute") or c.get("enabled") is False or c.get("is_valid") is False:
                continue
            enf = float(influence.get(c["name"], c["influence"]))
            if enf == 0.0:
                continue
            fn = getattr(self, "_c_" + c["type"].lower(), None)
            if fn is None:
                raise NotImplementedError(f"constraint {c['type']} on {owner}")
            space = c.get("owner_space", "WORLD")
            res = fn(c, self.to_space(owner, M, space), owner)
            if res is None:                                   # no valid target: nothing changes
                continue
            M = self.from_space(owner, res, space)
            if enf < 1.0:
                M = interp_m4(old, M, enf)
            old = M
        return M

    def _c_child_of(self, c, M, owner):
        T = self.target(c)
        if T is None:
            return None
        if not all(c.get(f"use_{k}_{a}", True) for k in ("location", "rotation", "scale") for a in "xyz"):
            raise NotImplementedError("Child Of with filtered channels (the rig has none)")
        return T @ _A(c["inverse_matrix"]) @ M

    def _c_damped_track(self, c, M, owner):
        T = self.target(c)
        if T is None:
            return None
        tar = T[:3, 3] - M[:3, 3]
        if np.linalg.norm(tar) == 0:
            return M
        tar = tar / np.linalg.norm(tar)
        ax, sign = TRACK[c["track_axis"]]
        ob = M[:3, ax] * sign
        ob = ob / np.linalg.norm(ob) if np.linalg.norm(ob) else np.eye(3)[ax] * sign
        raxis = np.cross(ob, tar)
        rangle = math.acos(max(-1.0, min(1.0, float(ob @ tar))))
        norm = float(np.linalg.norm(raxis))
        if norm < 1.1920929e-07:
            if abs(rangle) < math.pi - 0.01:
                return M
            rangle = math.pi
            ax2, s2 = TRACK[TRACK_ORDER[(TRACK_ORDER.index(c["track_axis"]) + 1) % 6]]
            raxis = np.cross(ob, M[:3, ax2] * s2)
            if np.linalg.norm(raxis) == 0:
                return M
        elif norm < 0.1:
            rangle = math.pi - math.asin(norm) if rangle > math.pi / 2 else math.asin(norm)
        out = M.copy()
        out[:3, :3] = axis_angle_to_mat3(raxis, rangle) @ M[:3, :3]
        return out

    def _c_limit_rotation(self, c, M, owner):
        order = c["euler_order"] if c.get("euler_order", "AUTO") != "AUTO" else self.rot_order(owner)
        loc, size = M[:3, 3].copy(), mat4_to_size(M)
        e = mat_to_eul(M, order)
        for i, a in enumerate("xyz"):
            if c[f"use_limit_{a}"]:
                lo, hi = c[f"min_{a}"], c[f"max_{a}"]
                e[i] = (min(max(e[i], lo), hi) if c.get("use_legacy_behavior") else _clamp_angle(e[i], lo, hi))
        return trs(loc, eul_to_mat3(e, order), size)

    def _c_transform(self, c, M, owner):
        T = self.target(c)
        if T is None:
            return None
        frm, to = c["map_from"], c["map_to"]
        sfx = {"LOCATION": "", "ROTATION": "_rot", "SCALE": "_scale"}
        if frm == "ROTATION":
            mode = c.get("from_rotation_mode", "AUTO")
            dvec = mat_to_eul(T, mode if mode in ROT_ORDERS else self.rot_order(owner))
        elif frm == "SCALE":
            dvec = list(mat4_to_size(T))
        else:
            dvec = list(T[:3, 3])
        fmin = [c[f"from_min_{a}{sfx[frm]}"] for a in "xyz"]
        fmax = [c[f"from_max_{a}{sfx[frm]}"] for a in "xyz"]
        sval = []
        for i in range(3):
            if not c.get("use_motion_extrapolate"):
                dvec[i] = min(max(dvec[i], fmin[i]), fmax[i])
            sval.append((dvec[i] - fmin[i]) / (fmax[i] - fmin[i]) if fmax[i] - fmin[i] else 0.0)
        idx = {"X": 0, "Y": 1, "Z": 2}
        tmin = [c[f"to_min_{a}{sfx[to]}"] for a in "xyz"]
        tmax = [c[f"to_max_{a}{sfx[to]}"] for a in "xyz"]
        new = [tmin[i] + sval[idx[c[f"map_to_{a}_from"]]] * (tmax[i] - tmin[i]) for i, a in enumerate("xyz")]
        loc, rot, size = mat4_to_loc_rot_size(M)
        if to == "ROTATION":
            order = c["to_euler_order"] if c.get("to_euler_order", "AUTO") != "AUTO" else self.rot_order(owner)
            mix = c["mix_mode_rot"]
            if mix == "REPLACE":
                rot = eul_to_mat3(new, order)
            elif mix == "BEFORE":
                rot = eul_to_mat3(new, order) @ rot
            elif mix == "AFTER":
                rot = rot @ eul_to_mat3(new, order)
            else:                                                # ADD
                old = mat3_normalized_to_eul(rot, order)
                rot = eul_to_mat3([new[i] + old[i] for i in range(3)], order)
        elif to == "SCALE":
            size = np.array(new) * size if c["mix_mode_scale"] == "MULTIPLY" else np.array(new)
        else:
            loc = loc + np.array(new) if c["mix_mode"] == "ADD" else np.array(new)
        return trs(loc, rot, size)

    def _c_stretch_to(self, c, M, owner):
        """stretchto_evaluate (SWING_Y keep axis)."""
        T = self.target(c)
        if T is None:
            return None
        size = mat4_to_size(M)
        R = normalize_cols(M[:3, :3])
        dist = float(np.linalg.norm(M[:3, 3] - T[:3, 3])) / size[1]
        org = c["rest_length"] or dist
        scale = np.ones(3)
        scale[1] = dist / org
        bulge = (org / dist) ** c["bulge"] if dist else 1.0
        vol = c["volume"]
        if vol == "VOLUME_XZX":
            scale[0] = scale[2] = math.sqrt(bulge)
        elif vol == "VOLUME_X":
            scale[0] = bulge
        elif vol == "VOLUME_Z":
            scale[2] = bulge
        size = size * scale
        vec = T[:3, 3] - M[:3, 3]
        vec = vec / (np.linalg.norm(vec) or 1.0)
        if c["keep_axis"] != "SWING_Y":
            raise NotImplementedError(f"Stretch To keep axis {c['keep_axis']}")
        R = rotation_between_vecs(R[:, 1], vec) @ R
        return trs(M[:3, 3], R, size)


def _clamp_angle(angle, lo, hi):
    """Limit Rotation (non-legacy): clamp with wrap-around, as rotlimit_evaluate's clamp_angle."""
    if hi - lo >= 2 * math.pi:
        return angle
    if hi <= lo:
        return lo
    wrap = lambda x: (x + math.pi) % (2 * math.pi) - math.pi   # noqa: E731
    mn, mx = wrap(lo - angle), wrap(hi - angle)
    if mn < mx:
        return angle + min(max(0.0, mn), mx)
    if mx >= 0.0 or mn <= 0.0:
        return angle
    return angle + mx if abs(mx) < abs(mn) else angle + mn
