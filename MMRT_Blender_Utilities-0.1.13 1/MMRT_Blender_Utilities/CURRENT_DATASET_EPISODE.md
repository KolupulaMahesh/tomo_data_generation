# Export the current animation in the VLA dataset layout

This export matches the layout inspected in
`/home/leo/Documents/work/vla_dataset_generation/Dataset/full_sequence_corrected_001`.
It renders one episode from the current evaluated scene, including regional NLA
timing. It does not call batch generation or randomize any action or material.

1. Finish the joint checks/retiming and select your dataset camera. Review any
   remaining violations from a partial retiming result.
2. Leave all required recorder entries enabled: `tomo`, `tomo_left_hand`,
   `tomo_right_hand`, and any active-object channels such as the iPhone.
3. In **Synthesizer**, set **Output Root** to the dataset root (for example,
   `/home/leo/Documents/work/vla_dataset_generation/Dataset`).
4. Set **Prefix** and **Start Index**, for example `retimed_current` and `1`.
   This creates `retimed_current_001`. An existing episode is never overwritten.
5. Under **Current Animation → One Dataset Episode**, optionally enable
   **Also Save Scene Copy**, then click **Export Current Animation as Dataset Episode**.

Do not restore original timing before this export. The dedicated button always
records MMRT and renders MP4 + depth, regardless of batch Dry Run, Count, Render,
Rec MMRT, or randomization settings. It uses the current camera, render engine,
resolution, and effective FPS, and records/renders every frame of the same output
time range. Global Old/New remapping is included. It uses text YAML even if the
normal recorder is configured for binary, then restores that setting.

Example output:

```text
Dataset/
  metadata.py                         # preserves existing episodes, adds this one
  retimed_current_001/
    metadata.py                       # same core episode keys as generation batch
    retimed_current_001.mp4
    retimed_current_001.mmrt.yaml
    retimed_current_001.urdf
    retimed_current_001.srdf
    meshes/robot/...
    depth/depth_0000.exr ...           # actual output frame numbers; 32-bit EXR
    retimed_current_001.blend          # optional copy
```

URDF/SRDF and robot meshes come from the configured Recorder robots. Environment
outputs are also generated when the Recorder environment mode is enabled. The
original recording/environment settings, current frame, scene range, render
paths/formats, and compositor are restored after export, including on failure.
Animation actions, retimed NLA strips, and materials are not changed.

Existing root `metadata.py` is parsed as a literal dictionary; it is never executed.
Episode metadata uses `offsets_applied = {}` and `materials_swapped = []`, because
no new randomization is applied. It also records current-animation export mode,
retiming status, output/source ranges, and effective FPS. Prior batch metadata
entries remain intact.

Progress is printed to Blender's console. This is synchronous, like existing
batch generation, and Blender may be busy for the duration. If export fails,
partial files are retained for inspection, but the episode is not added to root
metadata. Use a different episode name for another attempt or move the incomplete
folder yourself. The source `.blend` is not overwritten.
