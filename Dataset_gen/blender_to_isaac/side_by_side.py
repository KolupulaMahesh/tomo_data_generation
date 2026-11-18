"""Side-by-side video of two packed episodes (rgb_jpg.npy + rgb_off.npy): left = ours, middle = the reference, right =
their difference x4, with the frame and step on top; prints how different the pictures are.

    python side_by_side.py <ours_episode> <reference_episode> <out.mp4> [label left] [label middle]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------- settings ----------------
TILE = (640, 360)                                       # each panel
STEPS = [(0, "READY"), (45, "PICK_PHONE"), (155, "PLACE_PHONE"), (291, "CLOSE_CARTON"), (506, "APPLY_TAPE"),
         (609, "FLIP_BOX"), (725, "PACK_IN_HOLDER"), (1058, "MOVE_BOX_OUT"), (1096, "RETURN_HOME")]


def frames(ep):
    a, o = np.load(ep / "rgb_jpg.npy", mmap_mode="r"), np.load(ep / "rgb_off.npy")
    return [cv2.imdecode(np.asarray(a[o[i]:o[i + 1]]), cv2.IMREAD_COLOR) for i in range(len(o) - 1)]


def main():
    ours, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    labels = sys.argv[4:6] if len(sys.argv) > 5 else ["rig in Isaac (ours)", "dataset fs_000 (stored motion)"]
    A, B = frames(ours), frames(ref)
    n = min(len(A), len(B))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (TILE[0] * 3, TILE[1]))
    diffs = []
    for i in range(n):
        a, b = cv2.resize(A[i], TILE, interpolation=cv2.INTER_AREA), cv2.resize(B[i], TILE, interpolation=cv2.INTER_AREA)
        d = cv2.absdiff(A[i], cv2.resize(B[i], (A[i].shape[1], A[i].shape[0])))
        diffs.append(float(d.mean()))
        dv = cv2.resize(np.clip(d.astype(np.int32) * 4, 0, 255).astype(np.uint8), TILE)
        step = [s for f, s in STEPS if f <= i][-1]
        for img, label in ((a, labels[0]), (b, labels[1]), (dv, "difference x4")):
            cv2.putText(img, f"{label} | frame {i} {step}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                        cv2.LINE_AA)
        vw.write(np.hstack([a, b, dv]))
    vw.release()
    rep = {"frames": n, "mean_abs_diff_per_frame_0_255": {"mean": round(float(np.mean(diffs)), 3),
                                                          "max": round(float(np.max(diffs)), 3),
                                                          "max_frame": int(np.argmax(diffs))}}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print(f"[side by side] {n} frames -> {out} | picture difference (0-255 per pixel): mean {rep['mean_abs_diff_per_frame_0_255']}")


if __name__ == "__main__":
    main()
