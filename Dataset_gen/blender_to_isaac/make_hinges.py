"""The carton's hinge table for the training's carton read-out, from the rig (2026-11-09): for each bone task.yaml names, its angle per frame
away from its most open pose. The training's progress = angle / the last frame's angle, so 0 = most open and 1 = as at the end of the job;
"most open" is the pose farthest from the last frame's, because a carton can start closed and open later (the new robot's job does).

    env/bin/python make_hinges.py <scene> <out.npz> <bone> [<bone> ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig import Rig  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMATURE = "Armature_Box1"


def angle(Ra, Rb):
    return float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def main():
    scene, out, bones = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    rig = Rig(HERE / "blender_export" / scene / "rig.json")
    frames = np.arange(rig.doc["frame_start"], rig.doc["frame_end"] + 1)
    R = {b: [] for b in bones}
    for f in frames:
        rig.evaluate(float(f))
        for b in bones:
            R[b].append(rig.pose(ARMATURE, b)[:3, :3].copy())
    table = {}
    for b in bones:
        end = R[b][-1]
        away = np.array([angle(r, end) for r in R[b]])
        ref = R[b][int(away.argmax())]
        a = np.array([angle(r, ref) for r in R[b]], np.float64)
        table[f"{b}_angle"] = a
        print(f"[hinges] {b}: most open at frame {int(frames[away.argmax()])}, {np.degrees(a.max()):.1f} deg from it at most, "
              f"last frame {np.degrees(a[-1]):.1f} deg")
    np.savez(out, **table)
    print(f"[hinges] {out}: {len(bones)} bones, {len(frames)} frames")


if __name__ == "__main__":
    main()
