# data_gen: failure and recovery episodes for the boxing task

Takes the boxing demo (TOMO Gidora, 17 MMRT subtasks, 1101 frames) and makes full-sequence episodes in which things go
wrong: the robot notices, recovers with IK to where things actually are, checks the result, and continues the task from
where it failed. Physics is off; the episodes are rendered in Isaac Sim by the existing `Dataset_gen/blender_to_isaac`.
Cosmos is not used anywhere.

## Flow

```
 demo frame ──► Execute stage ──► Verify ──ok──► next stage ─────────────────────────────────┐
                    │ (failure injected here)                                              │
                    ▼                                                                      │
             Failure happens (miss, slip, drop, misplace, flap part way, carton tips) │
                    ▼                                                                      │
             Detect   (scene's actual state: object vs hand, object vs target)             │
                    ▼                                                                      │
             Classify (not_grasped / dropped / off_table / outside_target / tilted / ...)  │
                    ▼                                                                      │
             Choose recovery (by type and attempt: regrasp -> regrasp level -> ...) ◄──┐   │
                    ▼                                                                │   │
             Execute recovery (IK to the object's observed pose)                     │   │
                    ▼                                                                │   │
             Verify ──fail, attempts left──► reassess ───────────────────────────────┘   │
                    │ fail, none left / off table ──► terminal: stop, go home (success: false)
                    ▼ ok                                                                 │
             Resume the original task at the demo frame where it failed ──────────────────┘
                                                                                         ▼
                                                              Final verification (validate.py)
```

## Files

| File | Job |
|---|---|
| `config/task_gidora.yaml` | The task: source files, hands, the 17 stages, each grasp (grip / release frames), carton parts, table |
| `config/failures.yaml` | **Every number that varies**: episode mix A-E, retries, reaction time, tolerances, each scenario's ranges |
| `data_gen/source.py` | Loads the demo: MMRT recording (robot + phone) or blender_to_isaac `motion.npz` (whole scene) |
| `data_gen/kin.py` | URDF FK (the brain's `fk.py`) and two-arm IK, clamped to the URDF joint limits every iteration |
| `data_gen/builder.py` | Writes the episode frame by frame: demo frames, IK moves, grips, held / free / falling objects |
| `data_gen/observe.py` | Ground truth vs what the robot can observe (joints, grip IO, object in the head camera's view) |
| `data_gen/detect.py` | Checks from actual poses (held, placed, folded, upright) and failure classification |
| `data_gen/scenarios.py` | Each failure: how it is caused, detected, which recovery per attempt, how it is checked |
| `data_gen/episode.py` | The state machine above; episode categories; terminal handling |
| `data_gen/validate.py` | Final checks: task end reached, every object at its final pose, nothing held, joint limits, no jumps |
| `data_gen/export.py` | Label files, preview video, and the Isaac `motion.npz` / `episode.json` for rendering |
| `data_gen/generate.py` | Command line: N episodes, optional preview / render |
| `data_gen/stats.py` | Failure counts and recovery success rates over a folder |
| `data_gen/inspect_motion.py` | Lists the carton bones that turn in a frame window (to fill `carton_parts`) |

## Commands

```bash
cd ~/Downloads/dataset_generation/data_gen

# from the MMRT recording (phone scenarios; no rendering)
python3 -m data_gen.generate --episodes 50 --out out/run1 --preview
python3 -m data_gen.generate --episodes 8 --category single --scenario phone_drop_transport --out out/drop --preview

# statistics: which failures, which recoveries work
python3 -m data_gen.stats out/run1

# full scene + Isaac render (after the Gidora Blender export, see "Not done yet")
cd ../Dataset_gen/blender_to_isaac && for s in $(seq 0 49); do env/bin/python make_motion.py gidora_packing --seed $s; done
cd ../../data_gen
python3 -m data_gen.inspect_motion ../Dataset_gen/blender_to_isaac/runs/gidora_packing/ep_0000/generic/motion.npz \
    --armature Armature_Box1 --frames 328 454          # fill carton_parts bones in config/task_gidora.yaml
python3 -m data_gen.generate --episodes 50 --motion-dir ../Dataset_gen/blender_to_isaac/runs/gidora_packing \
    --out out/gidora --render
```

Options: `--category normal|single|multi|unrecoverable` forces one category; `--scenario NAME` uses only that scenario.

## Episode output (`out/<run>/ep_NNNN/`)

| File | Contents | Model input? |
|---|---|---|
| `recovery_labels.npz` | per frame: `demo_frame`, `q` + `joint_names`, `relay`, `grip_right`, `grip_left`, `stage` (+`stage_names`), `phase` (normal / failure / detect / recovery / verify / resume / terminal), `failure` (+names), `recovery` (+names), `attempt`, `fps` | joints, grips, stage: yes; phase / failure / recovery: training targets |
| `sim_truth.npz` | per frame: every object's world pose | **No**: simulation-only, for checks and evaluation |
| `events.json` | ordered events: `failure_injected`, `failure_detected` (with truth and observable snapshots), `recovery_decision`, `regrasp_reach`, `recovery_verified`, `task_resumed`, `terminal_failure` | – |
| `validation.json` | `success` and every check with its numbers | – |
| `preview.mp4` | top-view sketch with the phase on a coloured bar | – |
| `isaac/`, `../dataset/fs_NNN/` | with `--render`: the rendered episode in the training format, plus the three label files | – |

Run folder: `episodes.json` (one row per episode) and `stats.json` (failures detected per type, tries and successes per
recovery strategy, the attempt each recovery succeeded on, terminal failures).

## Scenarios

| # | failures.yaml | Failure | Detection | Recovery (attempt 1 → 2 → 3) | Status |
|---|---|---|---|---|---|
| 1 | `phone_pick_miss` | fingers close 2.5-6 cm beside the phone | phone not following the hand | regrasp at observed pose → regrasp level → ... | **tested** |
| 1 | `phone_pick_slip` | lifted, slips back to the table | same | same | **tested** |
| 2 | `phone_drop_transport` | falls while carried: table, tray, or off the table | not held; off table → stop | locate and regrasp → level grip | **tested** |
| 3 | `phone_misplaced` | released 2-6 cm off target, turned, tilted up to 18° | pose vs tray target | pick and replace → level grip and replace | **tested** |
| 4 | `flap_not_folded` | one side flap stops part way | bone angle vs folded | refold only that flap with its own hand → slower | needs bone names |
| 5 | `lid_not_closed` | lid / tuck lip part way | same | refold only that part | needs bone names |
| 6 | `tape_not_lifted` | fingers come up without the strip | not following the hand | regrasp | needs motion.npz |
| 7 | `tape_drop_transport` | strip falls while carried | not held | locate and regrasp | needs motion.npz |
| 8 | `tape_misapplied` | stuck off target | pose vs target | pick and re-apply | needs motion.npz |
| 9 | `box_lift_miss` | one index finger misses, carton not lifted | not following the hands | regrasp both hands | needs motion.npz |
| 10 | `box_falls_on_release` | tips over when the left finger lets go | tilt vs standing pose | both hands restore it upright, release again at half speed, stays 0.5 s | needs motion.npz |
| 11 | `box_drop_out` | carton dropped while carried out (ReturnBox) | not held | locate and regrasp | needs motion.npz |

## Setup done (2026-11-16)

- MMRT add-on 0.1.13 installed and enabled in Blender 5.3 (`~/.config/blender/5.3/scripts/addons`).
- `gidora_packing/iPhone_Packing_Gidora_6_synth.blend`: a copy with the Synthesizer panel filled
  (`scenes/gidora_packing.synth.json`); the original's md5 is unchanged. `scenes/gidora_packing.yaml` points at it.
- Exported (Blender only reads): `blender_export/gidora_packing/rig.json`, `scene/scene.usdc`, `objects_export.npz`.
- `blender_to_isaac/env/` built from `requirements.txt`. Seeds 0-4 of the full scene made (`runs/gidora_packing/ep_000N`).

## Tested (2026-11-16)

- MMRT source: 32 phone episodes, 32/32; 20 mixed: normal 2/2, single 5/5, multi 6/6 succeeded, unrecoverable 7/7 failed.
- Full scene, all 12 scenarios x 5 seeds: 59/60 succeeded. The one left: the carton dropped at the front-left table corner,
  15 cm beyond the left hand's reach - the robot labels it `box_out_out_of_reach` and stops (success: false).
- Full scene, 15 mixed episodes (seeds 200-214): normal 1/1, single 6/6, multi 4/4 (up to 3 failures each) succeeded;
  unrecoverable 4/4 marked failed.
- The held check on the untouched demo: "held" on frames 138-279 (grip IO 132-275; it waits until the phone is 1 cm up).
- Isaac render of a pick-miss episode (8 frames across failure / recovery / resume) and packed to the training format.
  Robot bones from the joints vs the demo: 2.9e-7; on an IK frame the rendered camera and hand vs URDF FK: 0.0 m.
- Joint limits are checked on every frame of every episode (IK clamps to the URDF limits).

## Limitations / still to test

1. Physics is off: falls are computed (gravity, then rest on the surface), not simulated.
2. No full-length rendered recovery dataset yet (only the 8-frame render test). About 5 min per episode on the laptop.
3. Which hand folds each side flap is taken from the bone names (E4.L = left); not checked frame by frame.
4. The carton in the holder tips 20-55 deg (leaning in the slot); lying flat is not modelled.
5. The tape is moved as one piece when dropped (its bones keep the shape they had), it does not bend on the table.
6. The camera-visibility flag in `events.json` is a pinhole projection of the object's origin, not of its outline.
