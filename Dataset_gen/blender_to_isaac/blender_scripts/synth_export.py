"""
Answer sheet for the Isaac rig's Synthesizer (2026-10-29): the data team's Synthesizer code (MMRT ActionManager +
TransformManager, exactly as synth_episode.py drives it: the iPhone offset over frames 0-155, the right arm's IK control
following it with Offset Parent over 127-165) with GIVEN offsets, then export_rig.py on the synthesized scene, so the
curves after key insertion, every object's world matrix and every bone are Blender's own. Nothing is recorded or saved;
the .blend is untouched.

    blender -b <the .blend> --python blender_scripts/synth_export.py -- blender_export/synth/<scene>_<tag> DX DY DYAW_DEG
(needs Blender with the data team's MMRT add-on enabled; only reads the .blend)
"""
import importlib
import runpy
import sys
import time
from pathlib import Path

import bpy

out_dir, dx, dy, dyaw = sys.argv[sys.argv.index("--") + 1:][:4]
dx, dy, dyaw = float(dx), float(dy), float(dyaw)
anim = importlib.import_module("MMRT_Blender_Utilities.common.synthesis.animation")

PHONE, ARM = "iPhone", "Right_arm_joint7_ik_control.001"
objects = {
    PHONE: {"frame_min": 0, "frame_max": 155},
    ARM: {"frame_min": 127, "frame_max": 165, "parent_offset_object": PHONE},
}
offsets = {PHONE: ((dx, dy, 0.0), (0.0, 0.0, dyaw)), ARM: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}

t0 = time.time()
scene = bpy.context.scene
actions = anim.ActionManager(list(objects), is_dry_run=False, scene=scene)
transforms = anim.TransformManager(scene, objects)
actions.capture_baselines()
transforms.cache_baseline_transforms()
actions.assign_episode_actions("synth_export")
applied = transforms.evaluate_and_apply_offsets(None, episode_offsets=offsets)
print(f"OFFSETS {applied} ({time.time() - t0:.0f} s)", flush=True)

sys.argv = [sys.argv[0], "--", out_dir]
runpy.run_path(str(Path(__file__).with_name("export_rig.py")), run_name="__main__")
(Path(out_dir) / "synth_offsets.txt").write_text(f"dx {dx} dy {dy} dyaw_deg {dyaw}\n")
print(f"SYNTH EXPORT done in {time.time() - t0:.0f} s", flush=True)
