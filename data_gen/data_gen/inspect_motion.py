"""List the bones of an armature that turn inside a frame window (to fill task_*.yaml carton_parts), and the objects.

    python3 -m data_gen.inspect_motion <motion.npz> --armature Armature_Box1 --frames 328 454
"""
from __future__ import annotations

import argparse

import numpy as np

from .kin import angle_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("--armature", default=None)
    ap.add_argument("--frames", type=int, nargs=2, default=None)
    a = ap.parse_args()
    M = np.load(a.motion)
    print("objects that move:", ", ".join(str(x) for x in M["object_names"]))
    arms = sorted(set(str(x) for x in M["bone_owner"]))
    print("armatures:", ", ".join(arms))
    if not a.armature:
        return
    f0, f1 = a.frames or (0, len(M["frames"]) - 1)
    rows = []
    for k, (o, b) in enumerate(zip(M["bone_owner"], M["bone_names"])):
        if str(o) != a.armature:
            continue
        P = M["bone_pose"][f0:f1 + 1, k]
        turn = max(angle_deg(P[0][:3, :3], x[:3, :3]) for x in P[::2])
        if turn > 1:
            rows.append((turn, str(b)))
    for turn, b in sorted(rows, reverse=True):
        print(f"  {b:40s} turns {turn:6.1f} deg in {f0}..{f1}")


if __name__ == "__main__":
    main()
