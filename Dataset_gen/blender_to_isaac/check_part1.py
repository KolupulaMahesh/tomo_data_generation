"""Part 1 check: our F-curve evaluation (fcurves.py) vs Blender's own value of every F-curve at every frame
(fcurve_values.npz from export_rig.py), for each exported scene.

    python check_part1.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcurves import FCurve  # noqa: E402

# ---------------- settings ----------------
EXPORTS = Path(__file__).resolve().parent / "blender_export"
SCENES = ["iPhone_Packing_local", "iPhone_Packing_robot_safe"]
TOL = 1e-5                                 # Blender stores keys and values as float32


def curves_in_export_order(rig):
    """The F-curves in the order export_rig.py evaluated them: objects (bpy.data order), then other animated data."""
    out = []
    for name, o in rig["objects"].items():
        a = o["animation"]
        if a:
            out += [(name, f) for f in a["fcurves"]]
    for name, a in rig["other_animation"].items():
        out += [(name, f) for f in a["fcurves"]]
    return out


def main():
    for scene in SCENES:
        rig = json.loads((EXPORTS / scene / "rig.json").read_text())
        ref = np.load(EXPORTS / scene / "fcurve_values.npz")
        frames, values = ref["frames"], ref["values"]
        curves = curves_in_export_order(rig)
        assert len(curves) == len(values), (len(curves), len(values))
        worst, bad = [], 0
        for i, (owner, d) in enumerate(curves):
            assert d["data_path"] == str(ref["data_path"][i]) and owner == str(ref["owner"][i]), (owner, d["data_path"])
            fc = FCurve(d)
            ours = np.array([fc(float(f)) for f in frames])
            err = float(np.abs(ours - values[i]).max())
            worst.append((err, owner, d["data_path"], d["array_index"], int(frames[np.argmax(np.abs(ours - values[i]))])))
            bad += err > TOL
        worst.sort(reverse=True)
        print(f"[part 1] {scene}: {len(curves)} F-curves x {len(frames)} frames | max error {worst[0][0]:.2e} | "
              f"curves over {TOL:g}: {bad}")
        for w in worst[:5]:
            print(f"    {w[0]:.2e}  {w[1]} {w[2]}[{w[3]}] at frame {w[4]}")


if __name__ == "__main__":
    main()
