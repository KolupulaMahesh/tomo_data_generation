"""Fill a .blend's MMRT Synthesizer panel and save it as a COPY next to it (2026-11-09: the generator reads the random ranges from the
panel; a file whose panel is empty gives every episode the same motion). The original is not changed. The copy stays in the same folder so
its links to the other .blend files still work; open it in Blender to see or edit the entries in the panel.

    blender -b <scene>.blend --python blender_scripts/fill_synthesizer.py -- <copy>.blend scenes/<scene>.synth.json

The entries file (JSON: Blender's own Python has no YAML reader; missing ranges = 0):
    {"objects": [{"obj": "iPhone.001", "frame_min": 0, "frame_max": 132, "trans_min": [-0.15, -0.025, 0], "trans_max": [-0.025, 0.1, 0],
                  "rot_min_deg": [0, 0, -30], "rot_max_deg": [0, 0, 30]},
                 {"obj": "Right_arm_joint7_ik_control", "parent_offset_obj": "iPhone.001", "frame_min": 112, "frame_max": 145}]}
"""
import json
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out, entries = Path(argv[0]).resolve(), json.loads(Path(argv[1]).read_text())["objects"]
if out == Path(bpy.data.filepath).resolve():
    raise SystemExit("the copy must have another name: the original stays as it is")
if out.parent != Path(bpy.data.filepath).resolve().parent:
    raise SystemExit("the copy must stay in the same folder as the original (its links to the other .blend files are relative)")
scene = bpy.context.scene
syn = scene.mmrt_synthesizer
syn.objects.clear()
for e in entries:
    o = syn.objects.add()
    o.obj = scene.objects[e["obj"]]
    if e.get("parent_offset_obj"):
        o.parent_offset_obj = scene.objects[e["parent_offset_obj"]]
    o.frame_min, o.frame_max = int(e["frame_min"]), int(e["frame_max"])
    for k in ("trans_min", "trans_max", "rot_min_deg", "rot_max_deg"):
        setattr(o, k, tuple(float(v) for v in e.get(k, (0.0, 0.0, 0.0))))
    print(f"[synth] {o.obj.name}: frames {o.frame_min}-{o.frame_max}, x y z {list(o.trans_min)}..{list(o.trans_max)} m, "
          f"turn {list(o.rot_min_deg)}..{list(o.rot_max_deg)} deg, follows {o.parent_offset_obj.name if o.parent_offset_obj else None}", flush=True)
bpy.ops.wm.save_as_mainfile(filepath=str(out), copy=True, relative_remap=True)
print(f"[synth] saved {out}", flush=True)
