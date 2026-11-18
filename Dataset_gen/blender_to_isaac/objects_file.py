"""
objects_file.py - the generator writes the dataset's objects file itself (2026-11-08): every object's shape once per dataset, every
mesh's pose per frame per episode, in format tomo_objects_v1 (the same as w2ik_pipeline/w2ik/objects_file.py, whose writer this is a
frozen copy of: separate codebases). The YOLO pipeline (labels) and the W2 + IK pipeline (contacts, pieces) read it: everything the high-
level control is built from comes from files inside the dataset.

  scene  blender_export/<scene>/objects_export.npz (blender_scripts/export_scene_objects.py, once per scene, Blender read-only)
         -> <dataset>/scene_objects.npz: meshes, triangles, each mesh's vertices in its own frame, the bending of shape-changing meshes
            (the carton's flaps, a peeled sticker) per frame of the animation, vertex groups, markers
  poses  the episode's own motion (make_motion.py motion.npz: every moving object's world matrix per frame + every object's frame-0 one,
         randomised objects included) -> <dataset>/<episode>/object_poses.npz; a mesh the rig never moves keeps the export's pose
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FORMAT = "tomo_objects_v1"


def write_scene(path: Path, meshes: list, tri: dict, v_local: dict, d_local: dict, vg: dict, fps: float, frames: int, markers: list,
                scene: str):
    objects = sorted({m["object"] for m in meshes})
    info = {"format": FORMAT, "scene": scene, "fps": fps, "frames": frames, "meshes": meshes, "objects": objects, "markers": markers}
    arrays = {"info": np.array(json.dumps(info))}
    for i in range(len(meshes)):
        arrays[f"tri_{i}"] = tri[i]
        if i in v_local:
            arrays[f"v_{i}"] = v_local[i].astype(np.float32)
        if i in d_local:
            arrays[f"d_{i}"] = d_local[i].astype(np.float32)
        if i in vg:
            arrays[f"vg_{i}"] = vg[i]
    np.savez_compressed(path, **arrays)


def write_poses(path: Path, M: np.ndarray, source: str, d_local: dict | None = None):
    arrays = {"M": M.astype(np.float32), "info": np.array(json.dumps({"format": FORMAT, "source": source}))}
    for i, d in (d_local or {}).items():
        arrays[f"d_{i}"] = d.astype(np.float32)
    np.savez_compressed(path, **arrays)


def read_export(export: Path):
    """the scene export -> (meshes, frames, M (frames, meshes, 4, 4), tri, v_local, d_local (bending in the mesh's own frame), vg, info)."""
    z = np.load(export, allow_pickle=True)
    info = json.loads(str(z["info"]))
    meshes = info["objects"]
    for m in meshes:
        m["object"] = m["parents"][-1] if m["parents"] else m["name"]
    M = z["M"].astype(np.float64)
    tri = {i: z[f"tri_{i}"] for i in range(len(meshes))}
    v_local = {i: z[f"v_{i}"] for i in range(len(meshes)) if f"v_{i}" in z.files}
    vg = {i: z[f"vg_{i}"] for i in range(len(meshes)) if f"vg_{i}" in z.files}
    d_local = {}
    for i in range(len(meshes)):
        if f"vw_{i}" in z.files:                                     # world -> the mesh's own frame (the full matrix: scaled objects)
            vw = z[f"vw_{i}"].astype(np.float64)
            d_local[i] = np.einsum("fvi,fji->fvj", vw - M[:, i, None, :3, 3], np.linalg.inv(M[:, i, :3, :3]))
    return meshes, np.asarray(z["frames"]), M, tri, v_local, d_local, vg, info


def scene_file(export: Path, out: Path, scene: str):
    meshes, frames, M, tri, v_local, d_local, vg, info = read_export(export)
    write_scene(out, meshes, tri, v_local, d_local, vg, float(info.get("fps") or 30), len(frames), info.get("markers", []), scene)
    return len(meshes), len(d_local)


def episode_poses(export: Path, motion: Path, idx, out: Path) -> dict:
    """this episode's pose of every mesh at its frames idx (frame numbers) -> object_poses.npz; returns what came from where."""
    meshes, frames_e, M, _, _, d_local, _, _ = read_export(export)
    mo = np.load(motion, allow_pickle=True)
    idx = np.asarray(idx, int)
    rows_m = idx - int(mo["frames"][0])
    # a recovery episode (data_gen) is longer than the animation: each of its frames shows the scene of a demo frame
    ef = np.asarray(mo["demo_frames"])[rows_m] if "demo_frames" in mo.files else idx
    rows_e = ef - int(frames_e[0])
    moving = [str(n) for n in mo["object_names"]]
    W = mo["object_world"]
    w0n = [str(n) for n in mo["world0_names"]] if "world0_names" in mo.files else []
    Me = M[rows_e].copy()
    src = {"motion": [], "frame0": [], "export": []}
    for i, m in enumerate(meshes):
        if m["name"] in moving:
            Me[:, i] = W[rows_m, moving.index(m["name"])]
            src["motion"].append(m["name"])
        elif m["name"] in w0n:                                       # never moves in this episode, maybe randomised: its own place
            Me[:, i] = mo["world0"][w0n.index(m["name"])]
            src["frame0"].append(m["name"])
        else:
            src["export"].append(m["name"])
    # the bending per frame of this episode (the export covers the whole animation; an episode may stop earlier)
    d_ep = {i: d[rows_e] for i, d in d_local.items()} if len(rows_e) != len(frames_e) or (rows_e != np.arange(len(frames_e))).any() else None
    write_poses(out, Me, f"generator: motion.npz ({len(src['motion'])} meshes), frame 0 ({len(src['frame0'])}), export "
                         f"({len(src['export'])}: {src['export']})", d_ep)
    return src
