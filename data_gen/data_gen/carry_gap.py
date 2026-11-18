"""The demo's own carry, checked before an episode is made (user 2026-11-18: the carried phone stays >= 3 cm from the
carton, also in normal episodes). The phone's start place is random per seed (make_motion.py --seed), so the demo's
lift can pass the open lid closer than that; such a seed is skipped by generate_mix.sh.

    python3 -m data_gen.carry_gap --task <task yaml> --scene <scene> --motion <motion.npz>
exit 0 = every carried object keeps its carton_gap_m from grip until its place stage starts, 1 = too close (printed).
"""
import argparse
import json
import sys

import numpy as np
import yaml

from .export import B2I


def surface(scene, holder, per_tri=8):
    """the carton's mesh points per export frame (world, as the demo bends it): (frames, points (F, N, 3))."""
    Z = np.load(B2I / "blender_export" / scene / "objects_export.npz")
    info = json.loads(str(Z["info"]))
    ks = [i for i, o in enumerate(info["objects"]) if holder in o.get("armature", []) and f"vw_{i}" in Z.files]
    rng = np.random.default_rng(0)
    a, c = rng.random((2, per_tri, 1))
    m = a + c > 1
    a[m], c[m] = 1 - a[m], 1 - c[m]
    return [int(f) for f in Z["frames"]], Z, ks, a, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--motion", required=True)
    a = ap.parse_args()
    task = yaml.safe_load(open(a.task))
    M = np.load(a.motion)
    names = [str(n) for n in M["object_names"]]
    mframes = [int(f) for f in M["frames"]]
    holder = task["holder"]["object"]
    stages = dict((n, f) for n, f in task["stages"])
    efr, Z, ks, ta, tc = surface(a.scene, holder)
    bad = []
    for key, g in task["grasps"].items():
        need = g.get("carton_gap_m")
        if not need or not g.get("size_half") or g["object"] not in names:
            continue
        hs = np.array(g["size_half"])
        ctr = np.array(g.get("size_centre", [0, 0, 0]))
        W = M["object_world"][:, names.index(g["object"])]
        place0 = stages.get(g.get("place_stage"), g["release"])
        best = (1.0, -1)
        for f in range(int(g["grip"]), int(place0)):
            if f not in mframes or f not in efr:
                continue
            fi = efr.index(f)
            pts = []
            for k in ks:
                V = Z[f"vw_{k}"][fi].astype(float)
                T = Z[f"tri_{k}"]
                pts += [V, (V[T[:, 0]][:, None] * (1 - ta - tc)[None] + V[T[:, 1]][:, None] * ta[None]
                            + V[T[:, 2]][:, None] * tc[None]).reshape(-1, 3)]
            S = np.vstack(pts)
            L = (np.linalg.inv(W[mframes.index(f)]) @ np.c_[S, np.ones(len(S))].T).T[:, :3] - ctr
            d = float(np.linalg.norm(np.maximum(np.abs(L) - hs, 0), axis=1).min())
            if d < best[0]:
                best = (d, f)
        print(f"[carry_gap] {key}: closest to the carton {best[0] * 1000:.1f} mm at frame {best[1]} "
              f"(need {need * 1000:.0f} mm)", flush=True)
        if best[0] < need:
            bad.append(key)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
