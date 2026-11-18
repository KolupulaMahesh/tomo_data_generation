"""Synthesizer check: our synth.py on the base rig vs Blender's own Synthesizer for the same offsets
(blender_scripts/synth_export.py -> blender_export/synth/<scene>_<name>/: the re-keyed curves, their values, and the
evaluated trace). Per answer sheet:
  keys    the re-keyed curves (phone + right-arm IK control): key values, handles, interpolation vs Blender's
  values  those curves at every frame vs Blender's own evaluation
  motion  every robot joint (deg) and the phone / IK control / hand (mm, deg) vs Blender's trace

    python check_synth.py [scene] [first:last:step]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_part2 import EXPORTS, err  # noqa: E402
from check_part3 import ROBOT, joint_angles  # noqa: E402
from rig import Rig  # noqa: E402
from synth import PHONE_SETUP, Synthesizer, phone_offsets  # noqa: E402

# ---------------- settings ----------------
SCENE = sys.argv[1] if len(sys.argv) > 1 else "iPhone_Packing_local"
FRAMES = slice(*[int(x) for x in sys.argv[2].split(":")]) if len(sys.argv) > 2 else slice(0, 1116, 1)
WATCH = ["iPhone", "Right_arm_joint7_ik_control.001", "tomo_right_hand"]


def main():
    for sheet in sorted((EXPORTS / "synth").glob(f"{SCENE}_*")):
        t0 = time.time()
        v = (sheet / "synth_offsets.txt").read_text().split()
        dx, dy, dyaw = float(v[1]), float(v[3]), float(v[5])
        rig = Rig(EXPORTS / SCENE / "rig.json")                     # the BASE scene: our synthesizer re-keys it
        Synthesizer(rig, PHONE_SETUP).run(phone_offsets(dx, dy, dyaw))
        ans = json.loads((sheet / "rig.json").read_text())["objects"]
        kerr, ierr, nkeys, kworst = 0.0, 0, 0, ""
        for n in PHONE_SETUP:
            theirs = {(f["data_path"], f["array_index"]): f for f in ans[n]["animation"]["fcurves"]}
            for fc in rig.curves[n]:
                t = theirs[(fc.data_path, fc.index)]
                if len(t["keys"]) != len(fc.keys):
                    raise SystemExit(f"{sheet.name} {n} {fc.data_path}[{fc.index}]: {len(fc.keys)} keys vs Blender "
                                     f"{len(t['keys'])}")
                for a, b in zip(fc.keys, t["keys"]):
                    e = max(abs(x - y) for x, y in zip(a["co"] + a["handle_left"] + a["handle_right"],
                                                       b["co"] + b["handle_left"] + b["handle_right"]))
                    if e > kerr:
                        kerr, kworst = e, f"{n} {fc.data_path}[{fc.index}] @ {b['co'][0]:g}"
                    ierr += a["interpolation"] != b["interpolation"]
                    nkeys += 1
        fv = np.load(sheet / "fcurve_values.npz")
        verr = 0.0
        for i, (o, p) in enumerate(zip(fv["owner"], fv["data_path"])):
            if str(o) in PHONE_SETUP:
                fc = next(c for c in rig.curves[str(o)] if c.data_path == str(p) and c.index == int(fv["array_index"][i]))
                verr = max(verr, float(np.abs(np.array([fc(float(f)) for f in fv["frames"]]) - fv["values"][i]).max()))
        tr = np.load(sheet / "trace.npz")
        frames, all_frames, W = tr["frames"][FRAMES], tr["frames"], tr["world"]
        onames = [str(x) for x in tr["object_names"]]
        bown, bnames, B = [str(x) for x in tr["bone_owner"]], [str(x) for x in tr["bone_names"]], tr["bone_matrix"]
        linkages = {a for a in set(bown) if a.startswith("Ref_finger_Armature")}
        jerr, werr = (0.0, -1, ""), {}
        for f in frames:
            k = int(np.flatnonzero(all_frames == f)[0])
            inject = {(bown[i], bnames[i]): B[k, i] for i in range(len(bnames)) if bown[i] in linkages}
            rig.evaluate(float(f), inject=inject)
            ours = {b: rig.pose(ROBOT, b) for (arm, b) in rig.bone if arm == ROBOT}
            theirs = {bnames[i]: B[k, i].astype(np.float64) for i in range(len(bnames)) if bown[i] == ROBOT}
            ja, jb = joint_angles(rig, ours), joint_angles(rig, theirs)
            for b in ja:
                d = abs((ja[b] - jb[b] + 180) % 360 - 180)
                if d > jerr[0]:
                    jerr = (d, int(f), b)
            for n in WATCH:
                e = err(rig.world(n), W[k, onames.index(n)].astype(np.float64))
                if e[0] + e[1] > sum(werr.get(n, (-1, -1))[:2]):
                    werr[n] = (e[0], e[1], int(f))
        print(f"[synth] {sheet.name}: dx {dx * 100:+.1f} cm, dy {dy * 100:+.1f} cm, yaw {dyaw:+.1f} deg | "
              f"{time.time() - t0:.0f} s", flush=True)
        print(f"    keys {nkeys}: worst {kerr:.2e} ({kworst}), interpolation differs on {ierr} | curve values worst "
              f"{verr:.2e} | robot joints worst {jerr[0]:.4f} deg ({jerr[2]} @ {jerr[1]})")
        for n, w in werr.items():
            print(f"    {n:34s} {w[0]:.4f} mm {w[1]:.4f} deg (frame {w[2]})")


if __name__ == "__main__":
    main()
