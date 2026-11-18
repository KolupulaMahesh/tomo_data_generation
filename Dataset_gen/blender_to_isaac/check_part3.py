"""Part 3 check: the robot's IK (both arms + head) computed by our copy of Blender's legacy solver (ik.py) vs Blender's
own result (trace.npz). Only the visual finger-linkage armatures are still given Blender's pose. Reported per frame
range: every robot joint angle (deg) vs Blender's, the hand / camera bones (mm, deg), and everything downstream
(hands, phone, carton, tape - which follow the hands through Child Of).

    python check_part3.py [scene] [first:last:step]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_part2 import EXPORTS, err  # noqa: E402
from rig import Rig  # noqa: E402

# ---------------- settings ----------------
SCENE = sys.argv[1] if len(sys.argv) > 1 else "iPhone_Packing_local"
FRAMES = slice(*[int(x) for x in sys.argv[2].split(":")]) if len(sys.argv) > 2 else slice(0, 1116, 1)
ROBOT = "tomo"
DOWNSTREAM = ["Right_arm_endlink", "Left_arm_endlink.002", "tomo_right_hand", "tomo_left_hand", "iPhone",
              "Armature_Box1", "Ripcord_Armature.003", "Head_Camera"]
JOINT_TOL_DEG = 0.01


def joint_angles(rig, P):
    """Each robot bone's own rotation about its axis (deg), from armature-space pose matrices P[(bone)]."""
    out = {}
    for (arm, b), bd in rig.bone.items():
        if arm != ROBOT or b not in P:
            continue
        par = bd["parent"]
        base = (P[par] @ bd["offs"]) if par else bd["ml"]
        L = np.linalg.inv(base[:3, :3]) @ P[b][:3, :3]
        out[b] = float(np.degrees(np.arctan2(L[1, 0], L[0, 0])))          # rotation about the bone's Z
    return out


def main():
    t0 = time.time()
    rig = Rig(EXPORTS / SCENE / "rig.json")
    tr = np.load(EXPORTS / SCENE / "trace.npz")
    frames = tr["frames"][FRAMES]
    onames = [str(x) for x in tr["object_names"]]
    bown, bnames, B = [str(x) for x in tr["bone_owner"]], [str(x) for x in tr["bone_names"]], tr["bone_matrix"]
    W, all_frames = tr["world"], tr["frames"]                # read once (each tr[...] decompresses the file)
    linkages = {a for a in set(bown) if a.startswith("Ref_finger_Armature")}
    jerr, perr, worst, iters = {}, {}, {}, []
    for f in frames:
        k = int(np.flatnonzero(all_frames == f)[0])
        inject = {(bown[i], bnames[i]): B[k, i] for i in range(len(bnames)) if bown[i] in linkages}
        rig.evaluate(float(f), inject=inject)
        ours = {b: rig.pose(ROBOT, b) for (arm, b) in rig.bone if arm == ROBOT}
        theirs = {bnames[i]: B[k, i].astype(np.float64) for i in range(len(bnames)) if bown[i] == ROBOT}
        ja, jb = joint_angles(rig, ours), joint_angles(rig, theirs)
        for b in ja:
            d = abs((ja[b] - jb[b] + 180) % 360 - 180)
            if d > jerr.get(b, (-1, 0))[0]:
                jerr[b] = (d, int(f))
        for b in ("Right_arm_joint7", "Left_arm_joint7", "head_camera"):
            e = err(ours[b], theirs[b])
            if e[0] + e[1] > perr.get(b, (-1, -1, 0, 0))[0] + perr.get(b, (-1, -1, 0, 0))[1]:
                perr[b] = (*e, int(f))
        for n in DOWNSTREAM:
            e = err(rig.world(n), W[k, onames.index(n)].astype(np.float64))
            if e[0] + e[1] > worst.get(n, (-1, -1, 0, 0))[0] + worst.get(n, (-1, -1, 0, 0))[1]:
                worst[n] = (*e, int(f))
        iters.append(max(rig.ik_iterations.values()))
    n = len(frames)
    print(f"[part 3] {SCENE}: frames {frames[0]}..{frames[-1]} ({n}) in {time.time() - t0:.0f} s "
          f"({(time.time() - t0) / n * 1000:.0f} ms/frame) | IK iterations median {int(np.median(iters))}, max {max(iters)}")
    bad = {b: v for b, v in jerr.items() if v[0] > JOINT_TOL_DEG}
    print(f"  robot joints: worst {max(v[0] for v in jerr.values()):.4f} deg | over {JOINT_TOL_DEG} deg: {len(bad)} of {len(jerr)}")
    for b, v in sorted(jerr.items(), key=lambda kv: -kv[1][0])[:8]:
        print(f"    {b:18s} {v[0]:.4f} deg (frame {v[1]})")
    for b, v in perr.items():
        print(f"  bone {b:18s} {v[0]:.4f} mm {v[1]:.4f} deg (frame {v[3]})")
    for nme, v in worst.items():
        print(f"  {nme:24s} {v[0]:.4f} mm {v[1]:.4f} deg (frame {v[3]})")


if __name__ == "__main__":
    main()
