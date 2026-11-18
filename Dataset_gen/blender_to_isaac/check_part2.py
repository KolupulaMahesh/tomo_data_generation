"""Part 2 check: parents, attachments (Child Of) and constraints. rig.py evaluates every object and every pose bone at
every frame; Blender's own result (trace.npz from export_rig.py) is the answer. The armatures with IK (part 3: the
robot's arms + head, the visual finger linkages) are given Blender's pose (injected); everything downstream of them
(hands, phone, carton, tape, camera aim, every object on the robot) is computed.

    python check_part2.py [scene ...]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig import Rig  # noqa: E402

# ---------------- settings ----------------
EXPORTS = Path(__file__).resolve().parent / "blender_export"
SCENES = sys.argv[1:] or ["iPhone_Packing_local", "iPhone_Packing_robot_safe"]
STATIC_FRAMES = (0, 557, 1115)             # objects that never move: checked at these frames
POS_MM, ROT_DEG = 0.01, 0.01               # pass: within 0.01 mm and 0.01 deg of Blender
KEY = ["iPhone", "Armature_Box1", "Ripcord_Armature.003", "Ripcord_Sheet_Armature", "tomo_left_hand",
       "tomo_right_hand", "Left_arm_joint7_ik_control.001", "Right_arm_joint7_ik_control.001",
       "Left_arm_joint7_ik_target", "Right_arm_joint7_ik_target", "camera_target.001", "camera_arrows", "Head_Camera"]


def err(A, B):
    """position error (mm), rotation error (deg, of the normalised axes), scale error (relative)."""
    pos = float(np.linalg.norm(A[:3, 3] - B[:3, 3])) * 1000
    sa, sb = np.linalg.norm(A[:3, :3], axis=0), np.linalg.norm(B[:3, :3], axis=0)
    Ra, Rb = A[:3, :3] / np.where(sa == 0, 1, sa), B[:3, :3] / np.where(sb == 0, 1, sb)
    c = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)
    return pos, float(np.degrees(np.arccos(c))), float(np.max(np.abs(sa - sb) / np.maximum(sb, 1e-9)))


def main():
    for scene in SCENES:
        t0 = time.time()
        rig = Rig(EXPORTS / scene / "rig.json")
        tr = np.load(EXPORTS / scene / "trace.npz")
        frames, onames, W = tr["frames"], [str(x) for x in tr["object_names"]], tr["world"].astype(np.float64)
        bown, bnames, B = [str(x) for x in tr["bone_owner"]], [str(x) for x in tr["bone_names"]], tr["bone_matrix"]
        ik_arms = {a for a, o in rig.O.items() if o["type"] == "ARMATURE" and o["in_scene"]
                   and any(c["type"] == "IK" for pb in o["pose_bones"].values() for c in pb["constraints"])}
        inj_idx = [i for i, a in enumerate(bown) if a in ik_arms]
        own_idx = [i for i, a in enumerate(bown) if a not in ik_arms]
        moving = np.abs(W - W[:1]).max(axis=(0, 2, 3)) > 1e-7
        evaluated = [i for i, n in enumerate(onames) if rig.O[n]["in_view_layer"]]   # Blender skips the rest
        res = {}                                             # name -> worst (pos, rot, scale, frame)
        for k, f in enumerate(frames):
            inject = {(bown[i], bnames[i]): B[k, i] for i in inj_idx}
            rig.evaluate(float(f), inject=inject)
            objs = evaluated if f in STATIC_FRAMES else [i for i in evaluated if moving[i]]
            for i in objs:
                e = err(rig.world(onames[i]), W[k, i])
                r = res.get(("obj", onames[i]))
                if r is None or e[0] + e[1] > r[0] + r[1]:
                    res[("obj", onames[i])] = (*e, int(f))
            for i in own_idx:
                e = err(rig.pose(bown[i], bnames[i]), B[k, i].astype(np.float64))
                r = res.get(("bone", bown[i], bnames[i]))
                if r is None or e[0] + e[1] > r[0] + r[1]:
                    res[("bone", bown[i], bnames[i])] = (*e, int(f))
        bad = {k: v for k, v in res.items() if v[0] > POS_MM or v[1] > ROT_DEG}
        nobj = sum(k[0] == "obj" for k in res)
        print(f"\n[part 2] {scene}: {nobj} objects ({sum(moving[i] for i in evaluated)} moving, every frame; "
              f"{len(onames) - len(evaluated)} excluded from the view layer, not evaluated by Blender) + "
              f"{len(res) - nobj} bones (every frame) checked, IK armatures given: {sorted(ik_arms)[:3]}... | "
              f"FAILING (> {POS_MM} mm or {ROT_DEG} deg): {len(bad)} | {time.time() - t0:.0f} s")
        for name in KEY:
            v = res.get(("obj", name))
            if v:
                print(f"    {name:34s} max {v[0]:.4f} mm  {v[1]:.4f} deg  scale {v[2]:.1e}  (frame {v[3]})")
        for arm in ("Armature_Box1", "tomo_left_hand", "tomo_right_hand", "Ripcord_Armature.003", "Ripcord_Sheet_Armature"):
            vs = [v for k, v in res.items() if k[0] == "bone" and k[1] == arm]
            if vs:
                print(f"    bones of {arm:23s} ({len(vs):2d}) max {max(v[0] for v in vs):.4f} mm  "
                      f"{max(v[1] for v in vs):.4f} deg")
        for k, v in sorted(bad.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:15]:
            print(f"    FAIL {k}: {v[0]:.3f} mm {v[1]:.3f} deg scale {v[2]:.1e} at frame {v[3]}")


if __name__ == "__main__":
    main()
