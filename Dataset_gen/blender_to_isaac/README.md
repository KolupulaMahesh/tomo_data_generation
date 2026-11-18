# blender_to_isaac

The Blender packing scene's machinery rebuilt in Isaac Sim's Python: Blender's curve maths, object and bone hierarchy,
constraints, Blender's legacy IK and the data team's Synthesizer. Isaac computes the motion the way Blender does,
moves objects the way Blender does, and renders the episodes.

The folder is **self-contained** (split out 2026-10-29). Everything it reads is a frozen copy inside it, so changes
elsewhere (the GUI, the brain, the teammate folders, the Blender work folder) cannot break it. The only things used from
outside are the two installed programs: Isaac Sim (`~/isaacsim`) and, only for a re-export, Blender.

## Layout

| Path | What |
|---|---|
| `fcurves.py`, `fcurve_edit.py`, `blender_math.py`, `rig.py`, `ik.py`, `synth.py` | The engine: Blender's curves, key insertion + handle smoothing, rotation maths, objects / bones / constraints, legacy IK, the Synthesizer |
| `check_part1.py` … `check_part3.py`, `check_synth.py` | Proof against Blender's own results: curves, constraints, IK, Synthesizer |
| `make_motion.py` | The rig's motion (phone moved by hand, or `--seed N`: random offsets in the .blend's Synthesizer ranges + a random look) → `runs/<scene>/…/generic/motion.npz` + `episode.json` (and the frozen generator's motion file) |
| `generate.sh` | **A dataset**: N episodes → `datasets/<scene>_<date>/fs_NNN/` in the format the training reads, + `task.yaml` |
| `dataset_generator.sh` / `.py` | Its window: choose or add a Blender scene, episodes, length → Generate (`gui_theme.py`: a copy of the training window's look) |
| `render_episode.py` | The generic renderer: the scene's look + the rig's motion as one Isaac file (`episode.usdc`, open it in Isaac Sim and play the timeline), then every frame from the head camera |
| `pack_episode.py` | One rendered episode → the training's episode format (labels from the .blend's markers; optional live black bars) |
| `objects_file.py` | The dataset's objects file (2026-11-08, format `tomo_objects_v1`): `scene_objects.npz` once per dataset + `<episode>/object_poses.npz`, written by `pack_episode.py` from `blender_export/<scene>/objects_export.npz` |
| `make_task_file.py` | The dataset's `task.yaml` (cut to the steps the episodes contain when they stop early) |
| `compare_renders.py` | Two rendered episodes compared frame by frame (colour, depth) + side-by-side video |
| `scenes/<scene>.yaml` | What the .blend does not say (objects left out, light calibration, steps, vacuum IO pins, looks) + `<scene>.task.yaml` for the training |
| `render_in_isaac.sh` | The frozen generator on the packing scene: one episode (colour + depth + labels + H.264 preview) |
| `watch_live.sh` | Watch the motion live in the Isaac window |
| `side_by_side.py` | Comparison video: ours / the reference episode / difference |
| `blender_scripts/` | `export_rig.py` (the rig), `export_scene_usd.py` (the look, with Blender's own USD exporter), `export_scene_objects.py` (every object's shape and pose per frame, for the objects file; a frozen copy of `w2ik_pipeline/blender/export_scene.py`), `synth_export.py`: run inside Blender, they only **read** the .blend |
| `blender_export/` | The exported rigs (`rig.json`, Blender's own curve values and trace) + `synth/` (4 answer sheets from Blender's Synthesizer) |
| `renderer/` | The dataset's Isaac generator: `tools/` (batch renderer), `tools_live/` (live window), `isaac_base/` (scene), `robot_scene/`, `assets/` |
| `robot_limits/` | Joint-limit / touch check (`check_motion.py`) and the two joint fixes (`fit_dataset_to_limits.py`, `fit_dataset_contacts.py`), with the robot model and the brain's kinematics |
| `reference/fs_000/` | One dataset episode, for the comparison video |
| `runs/` | Outputs (motion files, rendered episodes, videos) |
| `env/` | This folder's own Python (`requirements.txt`) |

## Run

```bash
cd Dataset_gen/blender_to_isaac
bash render_in_isaac.sh iPhone_Packing_local                     # the original motion
bash render_in_isaac.sh iPhone_Packing_local -0.09 -0.038 -27.8  # phone moved dx, dy (m), yaw (deg)
bash watch_live.sh iPhone_Packing_local                          # live window (after make_motion.py)
env/bin/python check_synth.py iPhone_Packing_local               # prove the Synthesizer against Blender
env/bin/python robot_limits/check_motion.py runs/iPhone_Packing_local/episode   # limits + touches
```

Videos: `runs/<scene>/[synth_<dx>_<dy>_<dyaw>/]episode/preview_h264.mp4` (H.264: plays in VS Code and browsers).
One Isaac Sim at a time.

To rebuild the Python environment:

```bash
~/isaacsim/kit/python/bin/python3.12 -m venv env && env/bin/pip install -r requirements.txt
```

## Where the frozen copies came from (2026-10-29, all checked identical to their source when copied)

| Here | Source | md5 (first 12) |
|---|---|---|
| `blender_export/iPhone_Packing_local/rig.json` | `tomo_vla_trainer-main/vla_full_sequence/blender_work/rig_export/` | f4c9395257a4 |
| … exported from | `vla_full_sequence/blender_work/iPhone_Packing_local.blend`, Blender 5.2.2 LTS | .blend 123711af7899 |
| `blender_export/iPhone_Packing_robot_safe/rig.json` | same folder | 0c78b66bbcef |
| `blender_scripts/` | `vla_full_sequence/blender_work/scripts/` (paths in comments updated) | |
| `renderer/isaac_base/base.usdc` | `tomo_vla_trainer-main/finalfullsequncepraveen/isaac_base/` (identical in `oldact_newdata_share`) | 42a804da9dc3 |
| `renderer/isaac_base/base_motion.npz` | same | 38bc18654789 |
| `renderer/isaac_base/carton_*`, `contact_segments.json` | `tomo_vla_trainer-main/oldact_newdata_share/isaac_base/` | |
| `renderer/robot_scene/`, `renderer/assets/` | `finalfullsequncepraveen/` (identical in `oldact_newdata_share`) | robot USD 83d2590f91ed |
| `renderer/tools/` | `finalfullsequncepraveen/tools/` (unchanged) | fs_isaac_gen.py e416dba4b5cb |
| `renderer/tools_live/` | `oldact_newdata_share/tools/` (unchanged; the policy files are only for model tests) | fs_isaac_gen.py 306705fe5bcb |
| `robot_limits/` | `tomo_vla_trainer-main/robot_limits/` (usage lines → `env/`; `physics_replay_fixed.py` left out: it needs `isaac_real_env`) | limits yaml f758970edeb5 |
| `robot_limits/kinematics/` | `tomo_brain_inference/service/kinematics/` | fk.py ff4cb00e8ec8 |
| `robot_limits/source/brain_robot.yaml` | `tomo_brain_inference/config/robot.yaml` (speeds for `build_limits.py`) | 9c6394d45cda |
| `reference/fs_000/` | `tomo_vla_trainer-main/fullsequencedatasetoct27/fs_000` | |

Since the copy (2026-10-29 cleanup):
- `oldact_newdata_share` was renamed `4.3.0_model`.
- `finalfullsequncepraveen` was deleted. Its scene, robot and assets are identical in `oldact_fullseq_share` and
  `4.3.0_model`. Nothing here depends on either.

The robot scene file names 15 sub-files that do not resolve (e.g. `3/simple_room.usd`; there is no folder `3/`). It is
the same in the teammate's original, and the scene renders correctly without them.

## Proof that the split changed nothing (2026-10-29)

- No file or link in the folder points outside it (code paths, symlinks, and every file the USD scenes load).
- Curves 1.19e-07, IK ≤ 0.0001°, Synthesizer on all 4 answer sheets ≤ 0.0002° and 2258 identical keys: the same
  numbers as before the split.
- `make_motion.py`: the original and the moved-phone motion files are bit-for-bit identical to the ones made before.
- A full Isaac render from this folder vs the render made before the split:
  - joints, phone, carton, suction and all labels are identical;
  - depth is identical on every pixel of every frame;
  - colour is 1.5 grey levels apart on average.
  Two renders of the SAME setup are 1.56 grey levels apart (Isaac's ray-traced sampling noise), so the colour difference
  is noise, not the split.

## Generate a dataset (2026-10-29)

```bash
bash generate.sh iPhone_Packing_local 50                 # 50 whole episodes (seeds 0-49)
bash generate.sh iPhone_Packing_local 50 0 505 --bars    # up to the closed lid, live camera's black bars painted in
```

Or in its own window, `bash dataset_generator.sh` (TOMO Dataset Generator; separate from the training window): choose a
scene or add a Blender file (it runs both exports and writes the scene's settings file from a template), the number of
episodes and how long, then Generate; Stop ends the running step at once. Open the finished dataset in the training
window like any other. Per episode: the phone (every object the .blend's Synthesizer moves) at a random
offset inside the .blend's own ranges, the robot's motion re-solved by the rig, a random look (table colour, phone print
and hue), the real robot's limits checked (`limits.json`, a report: the demo itself breaks some), rendered with the live
head-camera lens (1920x1080, fx 1070.79, fy 1070.53, cx 962.31, cy 550.0), packed at 960x540. About 5 min per whole
episode on this laptop. The same seed always gives the same episode.

Labels: the vacuums are the robot's IO byte from the .blend's markers (`relay`; io0 = right = `suction`, io2 = left =
`left_suction`). In the packing scene the markers keep the right vacuum on at 594–607 as well, where the old dataset's
attach flags (`inf_pick | inf_box_out`) had it off. The attach flags are still in `labels.npz`.

Tested 2026-10-29: `datasets/test_g3` (2 episodes, seeds 0 and 1, up to the closed lid, live bars): the phone at two
different places, the hand reaching it there and placing it in the carton; vacuums right 155–289, left 492–505; the
training's own episode check (`tomo_vla_train_gui/isaac_format.py`) finds no problem. About 3 min per episode to frame
505.
The robot-limits report says the demo motion itself breaks the robot's limits in places (it did before the
randomisation too); the two joint fixes in `robot_limits/` are not applied by `generate.sh`.

## A new Blender file

What the .blend itself must follow: **[BLENDER_FILE_FORMAT.md](BLENDER_FILE_FORMAT.md)**.

1. Export it, with Blender reading the file only:
   `blender -b <new>.blend --python blender_scripts/export_rig.py -- blender_export/<scene>` (the rig) and
   `blender -b <new>.blend --python blender_scripts/export_scene_usd.py -- blender_export/<scene>` (the look).
   The objects export (`export_scene_objects.py -- blender_export/<scene>/objects_export.npz --urdf
   robot_limits/source/robot_description.urdf`) is run by the window and, when it is missing, by `generate.sh`.
2. Write `scenes/<scene>.yaml` (copy the packing one): objects to leave out, steps, which IO pin is which vacuum, the
   object labelled as the phone, the looks to randomise, the training's task file.
3. `bash generate.sh <scene> 2` and look at the episodes.

Refused with an error naming it: a constraint, rotation mode or parent type the rig does not have; drivers, NLA strips or
animated materials / lights / cameras (`export.json` → `unsupported_animation`); a mesh whose shape changes over time
other than by bones (`geometry_changes`, measured at 5 frames). The robot is always TOMO (`tomo` in the .blend → the
robot's own USD, `renderer/robot_scene/`).

## The generic renderer: proof (2026-10-29)

- The look export (Blender 5.2.2) vs the teammate's `base.usdc`: the same 3,200 prims, every mesh within 0.0013 mm at
  frame 0, the same 31 materials, 27 textures, 9 skeletons (joints, rest and bind poses identical). The prims' world
  poses = Blender's (2.6e-06 m); the skeleton animation convention = Blender's exporter's (parent pose⁻¹ @ pose).
- `episode.usdc` read back = the rig: objects 2.2e-08 m, bones 4e-05 (USD stores bone scale in half precision).
- Full packing episode, generic renderer vs the frozen generator (both with the timing fix below): colour 1.61 grey
  levels apart (two renders of the same setup: 1.56); depth differs on 0.018 % of the pixels (about 95 scattered pixels
  per frame), not yet explained.
- Labels rebuilt from the rig = the old dataset's: carton pose 2.2e-07, the five attach flags identical.
- **Timing bug found**: Replicator's `step()` (default `delta_time=None`) plays the timeline on before each capture.
  `render_episode.py` passes `delta_time=0.0`. The frozen generator does not: in the existing dataset the carton and
  tape (timeline-driven) show a later moment than the robot and phone (set directly). On the last frame the timeline
  loops back to frame 0 (seen in our first render; not yet checked in the dataset's).

## Not wired in yet

- The two joint fixes run after rendering. The Blender demo itself puts both shoulders at 19° against the servo's 14°
  limit (about 30 frames each) and has 3 touching part pairs; see `robot_limits/check_motion.py`.
- Moving objects other than the phone (carton, plate, holder). (Cosmos looks: removed 2026-11-18, not used.)
