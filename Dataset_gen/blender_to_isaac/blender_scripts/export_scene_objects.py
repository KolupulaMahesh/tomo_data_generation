# FROZEN COPY (2026-11-08) of w2ik_pipeline/blender/export_scene.py, md5 9432577dfb8f: the generator keeps its own copy (separate codebases).
# Used once per scene (dataset_generator.py "add scene", generate.sh if missing) -> blender_export/<scene>/objects_export.npz,
# from which pack_episode.py writes the dataset objects file (objects_file.py).
"""
export_scene.py - run INSIDE Blender, read-only. Everything the W2 + IK pipeline needs from the scene, for any .blend made in the
generator's format (Dataset_gen/blender_to_isaac/BLENDER_FILE_FORMAT.md):

  objects   every rendered mesh that is not the robot. The robot = a mesh named like one of the URDF's links (Blender's .001 /
            _piano suffixes dropped), parented to the rig's armatures (format 1.1: tomo, tomo_right_hand, tomo_left_hand), or in a
            collection that is mostly such meshes, or in a collection named in --robot-collections. (Checking for objects linked
            from the rig files does not work: the packing scene has the robot appended and the carton, stickers and jig linked.)
            Each object's surface as triangles; a part of a mesh (the carton's flaps) is part of that object
  poses     each object's world matrix per frame; for an object whose shape changes (bent by an armature: the carton's flaps,
            the peeled sticker) its surface in world coordinates per frame
  markers   the MMRT markers with their IO events (format 1.5)
The scene's world is the robot base frame (format 1.1).

    blender -b <scene>.blend --python blender/export_scene.py -- <out.npz> --urdf <robot.urdf> [--robot-collections A,B,...]
"""
import json
import re
import sys
import xml.etree.ElementTree as ET

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
out = argv[0]
extra_robot = set(argv[argv.index("--robot-collections") + 1].split(",")) if "--robot-collections" in argv else set()
URDF_LINKS = {l.get("name") for l in ET.parse(argv[argv.index("--urdf") + 1]).getroot().iter("link")}
sc = bpy.context.scene
RIG = {"tomo", "tomo_right_hand", "tomo_left_hand"} | {r.obj.name for r in getattr(getattr(sc, "mmrt_recorder", None), "robots", []) if r.obj}
f0, f1 = sc.frame_start, sc.frame_end
frames = list(range(f0, f1 + 1))


def base_name(n):
    n = re.sub(r"\.\d+$", "", n)
    return re.sub(r"_(piano|hull_\d+)$", "", n)


def robot_named(o):
    n = base_name(o.name)
    return n in URDF_LINKS or any(l.endswith(n) for l in URDF_LINKS if len(n) > 5)


coll_robot = {}
for c in bpy.data.collections:
    meshes = [o for o in c.objects if o.type == "MESH"]
    coll_robot[c.name] = c.name in extra_robot or (len(meshes) > 0 and sum(robot_named(o) for o in meshes) >= 0.5 * len(meshes))


def robot(o):
    """the robot: named like a URDF link, parented to the rig's armatures, or in a robot collection."""
    p = o
    while p is not None:
        if p.name in RIG or robot_named(p) or any(coll_robot.get(c.name, False) for c in p.users_collection):
            return True
        p = p.parent
    return False


cands = [o for o in sc.objects if o.type == "MESH" and not o.hide_render and len(o.data.vertices) and not robot(o)]
names = [o.name for o in cands]
print(f"[export] {len(cands)} objects: {names}", flush=True)

# which ones change shape: probe 40 frames
sc.frame_set(f0)
dg = bpy.context.evaluated_depsgraph_get()


def ev_mesh(o, dg):
    ev = o.evaluated_get(dg)
    m = ev.to_mesh()
    v = np.array([vv.co[:] for vv in m.vertices], np.float64)
    m.calc_loop_triangles()
    t = np.array([tri.vertices[:] for tri in m.loop_triangles], np.int32)
    ev.to_mesh_clear()
    return v, t


base = {o.name: ev_mesh(o, dg) for o in cands}
deform = {o.name: False for o in cands}
for f in frames[:: max(1, len(frames) // 40)]:
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    for o in cands:
        if not deform[o.name]:
            v, _ = ev_mesh(o, dg)
            deform[o.name] = v.shape != base[o.name][0].shape or float(np.abs(v - base[o.name][0]).max()) > 1e-4

M = np.zeros((len(frames), len(cands), 4, 4), np.float32)
VW = {o.name: [] for o in cands if deform[o.name]}
for k, f in enumerate(frames):
    sc.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    for i, o in enumerate(cands):
        Mw = np.array(o.matrix_world, np.float64)
        M[k, i] = Mw
        if deform[o.name]:
            v, _ = ev_mesh(o, dg)
            VW[o.name].append((v @ Mw[:3, :3].T + Mw[:3, 3]).astype(np.float32))
    if k % 100 == 0:
        print(f"[export] frame {f}", flush=True)


def chain(o):
    c = []
    while o.parent is not None:
        c.append(o.parent.name)
        o = o.parent
    return c


markers = []
for m in getattr(sc, "mmrt_markers", []) or []:
    ev = []
    for e in getattr(m, "events", []):
        ev.append({k: bool(getattr(e, k)) for k in (f"io{i}" for i in range(8)) if hasattr(e, k)})
    markers.append({"name": str(getattr(m, "name", "")), "frame": int(getattr(m, "frame", -1)), "events": ev})

arrays = {"frames": np.array(frames), "M": M}
info = []
for i, o in enumerate(cands):
    v, t = base[o.name]
    arrays[f"tri_{i}"] = t
    # each vertex's strongest vertex group (the carton's lid / flaps are regions of one mesh, moved by its bones): for part labels
    groups = [g.name for g in o.vertex_groups]
    if groups and len(o.data.vertices) == len(v):
        dom = np.full(len(v), -1, np.int16)
        for vv in o.data.vertices:
            if len(vv.groups):
                dom[vv.index] = max(vv.groups, key=lambda g: g.weight).group
        arrays[f"vg_{i}"] = dom
    if deform[o.name]:
        arrays[f"vw_{i}"] = np.stack(VW[o.name])
    else:
        arrays[f"v_{i}"] = v.astype(np.float32)                         # local coordinates; world = M[frame, i] @ v
    info.append({"name": o.name, "collection": [c.name for c in o.users_collection], "parents": chain(o),
                 "armature": [m.object.name for m in o.modifiers if m.type == "ARMATURE" and m.object],
                 "changes_shape": bool(deform[o.name]), "verts": int(len(v)), "triangles": int(len(t)),
                 "vertex_groups": [g.name for g in o.vertex_groups]})
arrays["info"] = np.array(json.dumps({"scene": bpy.data.filepath, "fps": sc.render.fps, "objects": info, "markers": markers}))
np.savez_compressed(out, **arrays)
print(f"[export] wrote {out}: {len(cands)} objects, {sum(deform.values())} change shape, {len(markers)} markers, {len(frames)} frames", flush=True)
