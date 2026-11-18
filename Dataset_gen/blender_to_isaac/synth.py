"""The data team's Synthesizer (MMRT_Blender_Utilities common/synthesis/animation.py: TransformManager
.cache_baseline_transforms + .evaluate_and_apply_offsets) on our rig (rig.py), so a moved object is re-keyed and the
motion re-solved the way Blender does it - not a playback of Blender's result.

Per frame of each target's range, in frame order:
  desired  = its original world matrix moved by its offset (location + offset, rotation = offset quat @ rotation), or,
             with an Offset Parent, the parent's desired matrix @ the original parent-to-target transform, then moved
  channels = the local location / rotation that give `desired` (BKE_object_apply_mat4, as the matrix_world setter; a
             full-influence Child Of stack is inverted first, as set_visual_matrix); targets with fewer constraints
             first, so a controller is set before the props that hang on it
  keys     = location + rotation keyed at that frame (fcurve_edit.insert_key)
then the handles are recalculated (Blender does it on every insert) and the keys inside [frame_min, frame_max) become
LINEAR. Everything downstream (IK targets, the arm IK, Child Of attachments) is the rig's own evaluation.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blender_math import mat3_normalized_to_eul, mat3_to_quat, mat4_to_loc_rot_size, quat_to_mat3, trs  # noqa: E402
from fcurve_edit import insert_key, recalc_handles  # noqa: E402

# the data team's setup, as stored in the .blend (rig.json scene_mmrt.mmrt_synthesizer) and found by reproducing their 5
# episodes exactly (tomo_vla_trainer-main/vla_full_sequence/blender_work/scripts/synth_episode.py)
PHONE_SETUP = {"iPhone": {"frame_min": 0, "frame_max": 155},
               "Right_arm_joint7_ik_control.001": {"frame_min": 127, "frame_max": 165, "parent_offset_object": "iPhone"}}


def phone_offsets(dx, dy, dyaw_deg):
    return {"iPhone": ((dx, dy, 0.0), (0.0, 0.0, dyaw_deg)), "Right_arm_joint7_ik_control.001": ((0, 0, 0), (0, 0, 0))}


def blend_setup(rig):
    """The Synthesizer's objects as any .blend stores them (scene_mmrt.mmrt_synthesizer.objects)."""
    return {o["obj"]: {"frame_min": o["frame_min"], "frame_max": o["frame_max"],
                       **({"parent_offset_object": o["parent_offset_obj"]} if o["parent_offset_obj"] else {})}
            for o in rig.doc["scene_mmrt"]["mmrt_synthesizer"]["objects"]}


def random_offsets(rig, rng):
    """One episode: each object's offset uniform in its .blend range (trans_min..trans_max m, rot_min..rot_max deg)."""
    return {o["obj"]: (tuple(float(v) for v in rng.uniform(o["trans_min"], o["trans_max"])),
                       tuple(float(v) for v in rng.uniform(o["rot_min_deg"], o["rot_max_deg"])))
            for o in rig.doc["scene_mmrt"]["mmrt_synthesizer"]["objects"]}


# ------------------------------------------------------------------ Blender maths used by the synthesizer
def canonical_quat(R):
    """mat3_normalized_to_quat_fast: W non-negative."""
    q = mat3_to_quat(R)
    return -q if q[0] < 0 else q


def mul_qt(a, b):
    return np.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                     a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                     a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
                     a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]])


def eul_to_quat(e):
    """eul_to_quat (XYZ)."""
    ci, cj, ch = (math.cos(v * 0.5) for v in e)
    si, sj, sh = (math.sin(v * 0.5) for v in e)
    cc, cs, sc, ss = ci * ch, ci * sh, si * ch, si * sh
    return np.array([cj * cc + sj * ss, cj * sc - sj * cs, cj * ss + sj * cc, cj * cs - sj * sc])


def offset_matrix(M, translation, rotation_deg):
    """animation.offset_matrix: Matrix.decompose, location + offset, offset quat @ rotation, same scale."""
    loc, R, size = mat4_to_loc_rot_size(M)
    q = mul_qt(eul_to_quat([math.radians(v) for v in rotation_deg]), canonical_quat(R))
    return trs(np.add(loc, translation), quat_to_mat3(q), size)


class Synthesizer:
    def __init__(self, rig, objects: dict):
        self.rig, self.config = rig, {k: dict(v) for k, v in objects.items()}
        self.all_frames = set()
        for spec in self.config.values():
            spec["frames"] = range(int(spec["frame_min"]), int(spec["frame_max"]) + 1)
            self.all_frames.update(spec["frames"])

    # ---------------------------------------------------------------- the matrix_world setter
    def _parent_matrix(self, name):
        o, rig = self.rig.O[name], self.rig
        if not o["parent"]:
            return np.eye(4)
        P = rig.world(o["parent"])
        if o["parent_type"] == "BONE" and o["parent_bone"]:
            B = rig.pose(o["parent"], o["parent_bone"]).copy()
            B[:3, 3] += B[:3, 1] * rig.bone[(o["parent"], o["parent_bone"])]["length"]
            P = P @ B
        return P @ rig.pinv[name]

    def _channels(self, name, M):
        """BKE_object_apply_mat4(use_compat=False, use_parent=True) + BKE_object_mat3_to_rot: the local location /
        rotation (in the object's rotation mode) that make the UNCONSTRAINED world matrix M."""
        o = self.rig.O[name]
        if any(abs(v) > 1e-12 for v in o["delta_rotation_euler"]) or \
                any(abs(v) > 1e-12 for v in np.subtract(o["delta_rotation_quaternion"], [1, 0, 0, 0])):
            raise NotImplementedError(f"{name}: delta rotation")
        loc, R, _ = mat4_to_loc_rot_size(np.linalg.inv(self._parent_matrix(name)) @ M)
        q = canonical_quat(R)
        ch = {"location": list(np.subtract(loc, o["delta_location"]))}
        if o["rotation_mode"] == "QUATERNION":
            # q and -q are the same turn, but keys are interpolated channel by channel: a key on the other side of its neighbours makes
            # the object spin the long way round (2026-11-09: the new robot's hand has W = 0.000 at the grab, so any negative turn flipped
            # it, and its wrist spun 180 deg). The key keeps the sign of the curve it replaces.
            ref = self.rig.props(name)["obj"]["rotation_quaternion"]
            ch["rotation_quaternion"] = list(-q if np.dot(q, ref) < 0 else q)
        else:
            ch["rotation_euler"] = mat3_normalized_to_eul(quat_to_mat3(q), o["rotation_mode"])
        return ch

    def _target_channels(self, name, frame, desired, overrides):
        """set_visual_matrix: invert a full Child Of stack (only that is allowed), then the matrix_world setter."""
        rig = self.rig.evaluate(frame, overrides=overrides)
        o, infl = rig.O[name], rig.props(name)["influence"]
        active = [c for c in o["constraints"] if not c.get("mute") and infl[c["name"]] > 1e-8]
        M = desired
        if active:
            for c in active:
                axes = all(c.get(f"use_{k}_{a}", False) for k in ("location", "rotation", "scale") for a in "xyz")
                if c["type"] != "CHILD_OF" or abs(infl[c["name"]] - 1.0) > 1e-6 or not axes:
                    raise ValueError(f"{name}: constraint {c['name']!r} cannot be inverted safely (frame {frame})")
            C = rig.world(name) @ np.linalg.inv(self._parent_matrix(name) @ rig.basis(name))
            M = np.linalg.inv(C) @ desired
        return self._channels(name, M)

    def _curve(self, name, path, index):
        for fc, d in zip(self.rig.curves.get(name, []), self.rig.O[name]["animation"]["fcurves"]):
            if fc.data_path == path and fc.index == index:
                return fc, d.get("auto_smoothing") or "NONE"
        raise NotImplementedError(f"{name}: no F-curve {path}[{index}] (new curves are not needed for this setup)")

    # ---------------------------------------------------------------- the run
    def run(self, offsets: dict):
        """offsets: {target: ((dx, dy, dz) m, (rx, ry, rz) deg)}. Re-keys the targets' curves in self.rig."""
        rig, frames = self.rig, sorted(self.all_frames)
        orig = {n: {} for n in self.config}
        for f in frames:                                       # cache_baseline_transforms
            rig.evaluate(float(f))
            for n in self.config:
                orig[n][f] = rig.world(n).copy()
        rel = {n: {f: np.linalg.inv(orig[s["parent_offset_object"]][f]) @ orig[n][f] for f in frames}
               for n, s in self.config.items() if s.get("parent_offset_object")}
        keyed = {}
        for f in frames:
            calc = {}

            def resolve(n):
                if n not in calc:
                    par = self.config[n].get("parent_offset_object")
                    base = resolve(par) @ rel[n][f] if par else orig[n][f]
                    calc[n] = offset_matrix(base, *offsets[n])
                return calc[n]

            names = [n for n, s in self.config.items() if f in s["frames"]]
            desired = {n: resolve(n) for n in names}
            names.sort(key=lambda n: len(rig.O[n]["constraints"]))
            overrides = {}
            for n in names:                                    # controllers first; checked below
                overrides[n] = self._target_channels(n, float(f), desired[n], overrides)
            rig.evaluate(float(f), overrides=overrides)
            e = max(float(np.abs(rig.world(n) - desired[n]).max()) for n in names)
            if e >= 1e-4:
                raise ValueError(f"cannot resolve constrained target poses at frame {f} (error {e:.2e})")
            for n in names:                                    # insert_transform_key: location + rotation
                for path, vals in overrides[n].items():
                    for i, v in enumerate(vals):
                        fc, smoothing = self._curve(n, path, i)
                        insert_key(fc.keys, f, v)
                        keyed[(n, path, i)] = (fc, smoothing)
        for (n, path, i), (fc, smoothing) in keyed.items():
            recalc_handles(fc.keys, fc.extrapolation, smoothing)
            s = self.config[n]
            for k in fc.keys:
                if s["frame_min"] <= k["co"][0] < s["frame_max"]:
                    k["interpolation"] = "LINEAR"
            fc.x = [k["co"][0] for k in fc.keys]
        rig.evaluate(0.0)
        return offsets
