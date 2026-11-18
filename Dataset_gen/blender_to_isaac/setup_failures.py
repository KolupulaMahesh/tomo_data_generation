#!/usr/bin/env python3
"""Failure / recovery setup for a scene (2026-11-17): the same packing job as gidora_packing (same steps and timing), with
its own object names and sizes. Writes data_gen/config/task_<scene>.yaml from data_gen/config/task_gidora.yaml with the
names replaced and the sizes measured from the scene's export, and links it in scenes/<scene>.yaml (recovery_task,
track). The drop areas (red / green), frames and everything else stay as in task_gidora.yaml.

    env/bin/python setup_failures.py <scene> --pick <object> --tape <object> --carton <armature>
    env/bin/python setup_failures.py <scene> --list          # the scene's objects to choose from
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rig import Rig  # noqa: E402

DG = HERE.parent.parent / "data_gen" / "config"
TEMPLATE = DG / "task_gidora.yaml"
OLD = {"pick": "iPhone.001", "tape": "Ripcord_Armature.003", "carton": "Armature_Box1"}


def candidates(scene):
    """top-level objects that can be picked: meshes and armatures without a parent (robot excluded)."""
    rig = json.loads((HERE / "blender_export" / scene / "rig.json").read_text())["objects"]
    robot = (yaml.safe_load((HERE / "scenes" / f"{scene}.yaml").read_text()) or {}).get("robot", {}).get("object")
    out = []
    for n, o in rig.items():
        if o.get("parent") or n == robot or o.get("type") not in ("MESH", "ARMATURE"):
            continue
        out.append(n)
    return sorted(out), rig


class Measure:
    def __init__(self, scene):
        d = HERE / "blender_export" / scene
        self.Z = np.load(d / "objects_export.npz")
        self.info = json.loads(str(self.Z["info"]))
        self.frames = [int(f) for f in self.Z["frames"]]
        self.rig = Rig(d / "rig.json")

    def box(self, name, frame):
        """(lo, hi) of the object's meshes (itself, its children, meshes it bends) in its own frame at that frame."""
        Z, fi = self.Z, self.frames.index(frame)
        self.rig.evaluate(float(frame))
        W = self.rig.world(name)
        pts = []
        for k, o in enumerate(self.info["objects"]):
            if o["name"] == name or name in o.get("parents", []) or name in o.get("armature", []):
                if f"vw_{k}" in Z.files:
                    pts.append(Z[f"vw_{k}"][fi])
                elif f"v_{k}" in Z.files:
                    V = Z[f"v_{k}"]
                    pts.append((Z["M"][fi, k] @ np.c_[V, np.ones(len(V))].T).T[:, :3])
        if not pts:
            raise SystemExit(f"{name}: no mesh of it in the objects export")
        P = np.vstack(pts)
        L = (np.linalg.inv(W) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        return L.min(0), L.max(0)


def detect(scene, names):
    """which object each vacuum grip moves, from the .blend's own MMRT markers: after each Suck_* marker the held
    object moves with the hand until the next Release_*. The distinct objects in grip order = pick box, tape, carton."""
    doc = json.loads((HERE / "blender_export" / scene / "rig.json").read_text())
    mk = sorted((m["frame"], (m.get("marker_name") or m["name"]))
                for m in (doc.get("scene_mmrt") or {}).get("mmrt_markers") or doc.get("markers", []))
    sucks = [(f, n) for f, n in mk if n.lower().startswith("suck")]
    rels = [f for f, n in mk if n.lower().startswith("release")]
    rig = Rig(HERE / "blender_export" / scene / "rig.json")
    cand = [n for n in names if "ik_control" not in n and "camera" not in n.lower() and "[" not in n]
    pos = {}

    def at(f):
        if f not in pos:
            rig.evaluate(float(f))
            pos[f] = {n: rig.world(n)[:3, 3].copy() for n in cand}
        return pos[f]
    found = []
    for f, n in sucks:
        r = min([x for x in rels if x > f] or [f + 40])
        a, b = at(f), at(min(r, f + 40))
        moved = max(cand, key=lambda o: np.linalg.norm(b[o] - a[o]))
        if np.linalg.norm(b[moved] - a[moved]) > 0.01 and moved not in found:
            found.append(moved)
    if len(found) < 3:
        raise SystemExit(f"found only {found} moving with the vacuum grips ({[n for _, n in sucks]}): set them with "
                         f"--pick --tape --carton")
    return dict(zip(("pick", "tape", "carton"), found[:3]))


def r4(v):
    return "[" + ", ".join(f"{float(x):.4f}" for x in v) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--pick")
    ap.add_argument("--tape")
    ap.add_argument("--carton")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--auto", action="store_true", help="find pick box / tape / carton from the vacuum markers")
    a = ap.parse_args()
    names, rig = candidates(a.scene)
    if a.list:
        print(json.dumps(names))
        return
    if a.auto:
        for k, v in detect(a.scene, names).items():
            if getattr(a, k) is None:
                setattr(a, k, v)
        print(f"[setup] found: pick box {a.pick}, tape {a.tape}, carton {a.carton}")
    for k in ("pick", "tape", "carton"):
        if getattr(a, k) not in rig:
            raise SystemExit(f"--{k} {getattr(a, k)!r} is not an object of {a.scene}")
    tpl = yaml.safe_load(TEMPLATE.read_text())
    m = Measure(a.scene)
    t = TEMPLATE.read_text()

    # sizes, measured where the template measured them (pick / tape lying at frame 0, the closed carton in the holder)
    lo, hi = m.box(a.pick, 0)
    half, ctr = (hi - lo) / 2, (hi + lo) / 2
    t = re.sub(r"(^  phone:.*?on_table_z: )[0-9.]+", lambda x: x.group(1) + f"{max(-lo[2], 0.002):.4f}", t, count=1, flags=re.S | re.M)
    t = re.sub(r"(^  phone:.*?size_half: )\[[^\]]*\]", lambda x: x.group(1) + r4(half), t, count=1, flags=re.S | re.M)
    lo_t, hi_t = m.box(a.tape, 0)
    half_t, ctr_t = (hi_t - lo_t) / 2, (hi_t + lo_t) / 2
    half_t[2] = max(half_t[2], 0.001)
    t = re.sub(r"(^  tape:.*?size_half: )\[[^\]]*\]", lambda x: x.group(1) + r4(half_t), t, count=1, flags=re.S | re.M)
    t = re.sub(r"(^  tape:.*?size_centre: )\[[^\]]*\]", lambda x: x.group(1) + r4(ctr_t), t, count=1, flags=re.S | re.M)
    lo_c, hi_c = m.box(a.carton, int(tpl["holder"]["check_frame"]))
    t = re.sub(r"carton_box_local: \{[^}]*\}", f"carton_box_local: {{min: {r4(lo_c)}, max: {r4(hi_c)}}}", t)

    # names (after the sizes: the regexes above find the blocks by their keys, not by the names)
    for k, old in OLD.items():
        t = t.replace(old, getattr(a, k))
    out = DG / f"task_{a.scene}.yaml"
    t = (f"# {a.scene}: made by Dataset_gen/blender_to_isaac/setup_failures.py from task_gidora.yaml (same job, same timing):\n"
         f"# pick {a.pick}, tape {a.tape}, carton {a.carton}; sizes measured from the scene's export.\n") + t
    out.write_text(t)

    # carton bones the failures move: must exist in this carton
    bones = set((rig[a.carton].get("bones") or {}).keys())
    missing = sorted({b for p in tpl["carton_parts"].values() for b in p["bones"]} - bones)

    # the scene file: recovery_task + track; same job = same robot and camera as gidora_packing (copied when missing)
    sp = HERE / "scenes" / f"{a.scene}.yaml"
    s = sp.read_text()
    have = yaml.safe_load(s) or {}
    ref = (HERE / "scenes" / "gidora_packing.yaml").read_text()
    for key in ("robot", "camera"):
        if not have.get(key):
            blk = re.search(rf"^{key}:.*?(?=^\S)", ref, flags=re.S | re.M).group(0)
            blk = re.sub(r"\.\./robot_limits", "../robot_limits", blk)
            s = re.sub(rf"^# ?{key}:.*$", "", s, flags=re.M) + "\n" + blk
    # the look: same job = same Isaac light and the same balanced table colours as gidora_packing (a new scene starts
    # from the template's dim light and no colours: dark pictures, black table). The phone prints only for the same
    # picked object (another box keeps its own print).
    refc = yaml.safe_load(ref)
    for key in ("light_gain", "dome"):
        if have.get(key) != refc.get(key):
            s = re.sub(rf"^{key}:.*$", f"{key}: {refc[key]}                   # = gidora_packing (setup_failures.py)",
                       s, count=1, flags=re.M)
    if not have.get("random_look"):
        rl = refc.get("random_look") or {}
        look = {"balanced": True}
        rec = {n: v for n, v in (rl.get("recolor") or {}).items() if n in rig}
        if rec:
            look["recolor"] = rec
        tex = {n: v for n, v in (rl.get("retexture") or {}).items() if n in rig and n == a.pick}
        if tex:
            look["retexture"] = tex
        blk = "random_look: " + json.dumps(look) + "   # = gidora_packing's balanced look (setup_failures.py)"
        s = re.sub(r"^random_look:.*$", blk, s, count=1, flags=re.M) if re.search(r"^random_look:", s, flags=re.M) \
            else s + "\n" + blk + "\n"
    rel = Path("../../../data_gen/config") / out.name
    for key, val in (("recovery_task", str(rel)), ("track", a.pick)):
        if re.search(rf"^{key}:", s, flags=re.M):
            s = re.sub(rf"^{key}:.*$", f"{key}: {val}", s, count=1, flags=re.M)
        else:
            s += f"\n{key}: {val}\n"
    sp.write_text(s)
    print(json.dumps({"task_file": str(out), "pick_size_cm": [round(2 * x * 100, 1) for x in half],
                      "tape_size_cm": [round(2 * x * 100, 1) for x in half_t],
                      "carton_size_cm": [round((h - l) * 100, 1) for l, h in zip(lo_c, hi_c)],
                      "missing_carton_bones": missing}))
    if missing:
        print(f"WARNING: the carton {a.carton} has no bones {missing}: flap / lid failures are skipped for it")


if __name__ == "__main__":
    main()
