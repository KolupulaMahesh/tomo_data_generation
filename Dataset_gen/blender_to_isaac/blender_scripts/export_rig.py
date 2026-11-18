"""
Read-only export of the packing scene's RIG - everything Blender uses to compute the motion - so it can be rebuilt in
Isaac Sim and checked against Blender itself (2026-10-28, Praveen: "the Isaac version reproducing the dataset must be
using the way Blender works"). The .blend is NOT saved or changed.

  rig.json            scene (fps, frames, markers), every object (parent, parent bone, parent inverse, rest / frame-0
                      transforms), every constraint (object and bone, all settings incl. Child Of inverse matrices and
                      IK target / pole / chain), armature bones (rest matrices, parents, connect / inherit flags), pose
                      bone IK settings (locks, limits, stiffness), the armatures' IK solver, every animation: actions
                      (F-curves with every key: value, interpolation, handles, easing), drivers, NLA; the MMRT recorder and
                      synthesizer settings and the objects' MMRT properties
  fcurve_values.npz   Blender's own evaluation of every F-curve at every frame (checks our curve maths)
  trace.npz           Blender's evaluated result at every frame: every object's world matrix and every pose bone's
                      armature-space matrix (checks our constraints / IK / bones end to end)

    blender -b <scene>.blend --python blender_scripts/export_rig.py -- blender_export/<scene>     (only reads the .blend)
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np

OUT = Path(sys.argv[sys.argv.index("--") + 1:][0])
MAX_DEPTH = 4                                      # nesting of MMRT property groups


def conv(v, depth=0):
    """A Blender value -> JSON (ID -> its name, matrices / vectors -> lists, sets -> sorted lists)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, bpy.types.ID):
        return v.name_full                         # name [library]: linked objects share names
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    if hasattr(v, "bl_rna") and depth < MAX_DEPTH and not isinstance(v, bpy.types.ID):
        return rna(v, depth + 1)
    try:
        seq = list(v)
    except TypeError:
        return str(v)
    return [conv(x, depth) for x in seq]


def rna(struct, depth=0, skip=()):
    """Every RNA property of a struct (nested property groups / collections down to MAX_DEPTH)."""
    out = {}
    for p in struct.bl_rna.properties:
        k = p.identifier
        if k == "rna_type" or k in skip:
            continue
        try:
            v = getattr(struct, k)
        except Exception:
            continue
        if p.type == "POINTER":
            out[k] = v.name_full if isinstance(v, bpy.types.ID) else (rna(v, depth + 1) if v is not None and depth < MAX_DEPTH else None)
        elif p.type == "COLLECTION":
            if depth < MAX_DEPTH:
                out[k] = [conv(x, depth + 1) if not isinstance(x, bpy.types.ID) else x.name_full for x in v]
        else:
            out[k] = conv(v, depth)
    return out


def mat(m):
    return [list(map(float, row)) for row in m]


def fcurves_of(action, slot=None):
    """F-curves of an action: the legacy list, or (Blender 4.4+ layered actions) the channel bag of the slot."""
    if hasattr(action, "fcurves"):
        try:
            return list(action.fcurves)
        except Exception:
            pass
    out = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                if slot is None or bag.slot == slot:
                    out.extend(bag.fcurves)
    return out


def dump_fcurve(fc):
    return {"data_path": fc.data_path, "array_index": fc.array_index, "group": fc.group.name if fc.group else None,
            "extrapolation": fc.extrapolation, "mute": fc.mute, "auto_smoothing": getattr(fc, "auto_smoothing", None),
            "modifiers": [rna(m) for m in fc.modifiers],
            "keys": [{"co": list(k.co), "interpolation": k.interpolation, "easing": k.easing,
                      "handle_left": list(k.handle_left), "handle_right": list(k.handle_right),
                      "handle_left_type": k.handle_left_type, "handle_right_type": k.handle_right_type,
                      "back": k.back, "amplitude": k.amplitude, "period": k.period, "type": k.type}
                     for k in fc.keyframe_points]}


def dump_anim(idb, curves):
    ad = idb.animation_data
    if ad is None:
        return None
    slot = getattr(ad, "action_slot", None)
    d = {"action": ad.action.name_full if ad.action else None, "slot": getattr(slot, "identifier", None),
         "action_blend_type": ad.action_blend_type, "action_influence": ad.action_influence, "nla": [], "fcurves": [],
         "drivers": []}
    if ad.action:
        for fc in fcurves_of(ad.action, slot):
            d["fcurves"].append(dump_fcurve(fc))
            curves.append((idb, fc))
    for tr in ad.nla_tracks:
        d["nla"].append({"name": tr.name, "mute": tr.mute, "is_solo": tr.is_solo,
                         "strips": [{"name": s.name, "action": s.action.name_full if s.action else None,
                                     "frame_start": s.frame_start, "frame_end": s.frame_end,
                                     "action_frame_start": s.action_frame_start, "action_frame_end": s.action_frame_end,
                                     "blend_type": s.blend_type, "influence": s.influence, "mute": s.mute,
                                     "extrapolation": s.extrapolation, "repeat": s.repeat, "scale": s.scale}
                                    for s in tr.strips]})
    for fc in ad.drivers:
        dr = fc.driver
        d["drivers"].append({**dump_fcurve(fc), "driver": {
            "type": dr.type, "expression": dr.expression, "is_simple_expression": getattr(dr, "is_simple_expression", None),
            "use_self": dr.use_self,
            "variables": [{"name": v.name, "type": v.type,
                           "targets": [{"id": t.id.name_full if t.id else None, "data_path": t.data_path,
                                        "bone_target": t.bone_target, "transform_type": t.transform_type,
                                        "transform_space": t.transform_space, "rotation_mode": t.rotation_mode}
                                       for t in v.targets]} for v in dr.variables]}})
    return d


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    blend = Path(bpy.data.filepath)
    f0, f1 = scene.frame_start, scene.frame_end
    scene.frame_set(f0)
    curves = []                                    # (owner, fcurve) for the per-frame evaluation

    in_scene = {o.as_pointer() for o in scene.objects}
    in_vl = {o.as_pointer() for o in bpy.context.view_layer.objects}   # excluded collections are never evaluated
    objects = {}
    for ob in bpy.data.objects:
        o = {"type": ob.type, "parent": ob.parent.name_full if ob.parent else None, "parent_type": ob.parent_type,
             "parent_bone": ob.parent_bone or None, "matrix_parent_inverse": mat(ob.matrix_parent_inverse),
             "matrix_basis_f0": mat(ob.matrix_basis), "matrix_world_f0": mat(ob.matrix_world),
             "rotation_mode": ob.rotation_mode, "hide_render": ob.hide_render, "hide_viewport": ob.hide_viewport,
             **{k: list(getattr(ob, k)) for k in ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle",
                                                  "scale", "delta_location", "delta_rotation_euler",
                                                  "delta_rotation_quaternion", "delta_scale")},   # frame-0 values
             "in_scene": ob.as_pointer() in in_scene, "in_view_layer": ob.as_pointer() in in_vl, "data": ob.data.name_full if ob.data else None,
             "library": ob.library.filepath if ob.library else None,
             "constraints": [{"type": c.type, **rna(c)} for c in ob.constraints],
             "custom_props": {k: conv(ob[k]) for k in ob.keys() if not k.startswith("_")},
             "mmrt": {p.identifier: conv(getattr(ob, p.identifier)) for p in ob.bl_rna.properties
                      if p.identifier.startswith(("mmrt", "tomo"))},
             "animation": dump_anim(ob, curves)}
        if ob.type == "ARMATURE":
            o["ik_solver"] = ob.pose.ik_solver
            o["ik_param"] = rna(ob.pose.ik_param) if ob.pose.ik_param else None
            o["bones"] = {b.name: {"parent": b.parent.name if b.parent else None, "matrix_local": mat(b.matrix_local),
                                   "head_local": list(b.head_local), "tail_local": list(b.tail_local),
                                   "use_connect": b.use_connect, "use_inherit_rotation": b.use_inherit_rotation,
                                   "inherit_scale": b.inherit_scale, "use_local_location": b.use_local_location,
                                   "use_deform": b.use_deform, "use_relative_parent": b.use_relative_parent} for b in ob.data.bones}
            o["pose_bones"] = {pb.name: {
                "rotation_mode": pb.rotation_mode, "matrix_basis_f0": mat(pb.matrix_basis), "matrix_f0": mat(pb.matrix),
                **{k: list(getattr(pb, k)) for k in ("location", "rotation_quaternion", "rotation_euler",
                                                     "rotation_axis_angle", "scale")},              # frame-0 values
                **{k: conv(getattr(pb, k)) for k in ("lock_ik_x", "lock_ik_y", "lock_ik_z", "use_ik_limit_x",
                                                     "use_ik_limit_y", "use_ik_limit_z", "ik_min_x", "ik_max_x",
                                                     "ik_min_y", "ik_max_y", "ik_min_z", "ik_max_z", "ik_stiffness_x",
                                                     "ik_stiffness_y", "ik_stiffness_z", "ik_stretch",
                                                     "use_ik_linear_control", "use_ik_rotation_control",
                                                     "lock_location", "lock_rotation", "lock_scale")},
                "constraints": [{"type": c.type, **rna(c)} for c in pb.constraints],
                "custom_props": {k: conv(pb[k]) for k in pb.keys() if not k.startswith("_")},
                "mmrt": {p.identifier: conv(getattr(pb, p.identifier)) for p in pb.bl_rna.properties
                         if p.identifier.startswith(("mmrt", "tomo"))}} for pb in ob.pose.bones}
        objects[ob.name_full] = o
    other_anim = {}                                # animated non-object data (armatures, materials, scene, ...)
    for coll in (bpy.data.armatures, bpy.data.materials, bpy.data.cameras, bpy.data.lights, bpy.data.scenes,
                 bpy.data.shape_keys, bpy.data.node_groups):
        for idb in coll:
            a = dump_anim(idb, curves) if getattr(idb, "animation_data", None) else None
            if a and (a["fcurves"] or a["drivers"] or a["nla"]):
                other_anim[f"{type(idb).__name__}:{idb.name_full}"] = a

    mmrt_scene = {p.identifier: conv(getattr(scene, p.identifier)) for p in scene.bl_rna.properties
                  if p.identifier.startswith(("mmrt", "tomo"))}
    doc = {"blend": str(blend), "blend_md5": hashlib.md5(blend.read_bytes()).hexdigest(),
           "blender": bpy.app.version_string, "fps": scene.render.fps, "fps_base": scene.render.fps_base,
           "frame_start": f0, "frame_end": f1, "unit_scale": scene.unit_settings.scale_length,
           "markers": [{"name": m.name, "frame": m.frame} for m in scene.timeline_markers],
           "actions": [a.name_full for a in bpy.data.actions], "objects": objects, "other_animation": other_anim,
           "scene_mmrt": mmrt_scene}
    (OUT / "rig.json").write_text(json.dumps(doc, indent=1, default=str))
    print(f"EXPORT rig.json: {len(objects)} objects, {sum(o['type'] == 'ARMATURE' for o in objects.values())} armatures, "
          f"{len(curves)} F-curves, {len(doc['markers'])} markers ({time.time() - t0:.0f} s)", flush=True)

    # Blender's own evaluation, every frame: F-curves, object world matrices, pose bone matrices
    frames = np.arange(f0, f1 + 1)
    fc_vals = np.zeros((len(curves), len(frames)), np.float64)
    for i, (_, fc) in enumerate(curves):
        fc_vals[i] = [fc.evaluate(float(f)) for f in frames]
    np.savez_compressed(OUT / "fcurve_values.npz", frames=frames, values=fc_vals,
                        owner=np.array([o.name_full for o, _ in curves]), data_path=np.array([fc.data_path for _, fc in curves]),
                        array_index=np.array([fc.array_index for _, fc in curves]))
    obs = list(scene.objects)
    arms = [ob for ob in obs if ob.type == "ARMATURE"]
    bone_keys = [(a.name_full, pb.name) for a in arms for pb in a.pose.bones]
    W = np.zeros((len(frames), len(obs), 4, 4), np.float32)
    B = np.zeros((len(frames), len(bone_keys), 4, 4), np.float32)
    for k, f in enumerate(frames):
        scene.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        for i, ob in enumerate(obs):
            W[k, i] = np.array(ob.evaluated_get(dg).matrix_world)
        j = 0
        for a in arms:
            ae = a.evaluated_get(dg)
            for pb in a.pose.bones:
                B[k, j] = np.array(ae.pose.bones[pb.name].matrix)
                j += 1
        if f % 200 == 0:
            print(f"  trace frame {f}/{f1} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(OUT / "trace.npz", frames=frames, object_names=np.array([o.name_full for o in obs]), world=W,
                        bone_owner=np.array([a for a, _ in bone_keys]), bone_names=np.array([b for _, b in bone_keys]),
                        bone_matrix=B)
    print(f"EXPORT done -> {OUT}: trace {len(frames)} frames x {len(obs)} objects + {len(bone_keys)} bones "
          f"({time.time() - t0:.0f} s)", flush=True)


main()
