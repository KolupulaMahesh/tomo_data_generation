# TOMO robot limits for the packing animation

The robot model learns to move **exactly like the animation**. When the animation does something the real robot cannot do, the robot copies it, and the robot's controller stops the run. That happened on 2026-10-25:

- The shoulders were asked to go past their limit, so the controller refused the motion.
- Two left-hand fingers came too close, so the controller stopped the run with "collision".

All 5 recordings (`full_sequnece_001` … `005`) come from the same animation and have the same problems. This folder has the robot's real limits, the list of what to fix (with frame numbers), and a checker to test a new recording before generating data from it.

## What is in this folder

| File | What it is |
|---|---|
| `README.md` | This page: the rules, what to fix, how to check |
| `tomo_re_uhand_limits.yaml` | Every joint's real limits, the speed limits and the collision rule. Machine-readable, with the same joint names as your MMRT recordings |
| `report_full_sequnece_001.txt` / `.json` | Everything wrong in `full_sequnece_001`, with frame numbers. The other 4 recordings are the same |
| `check_motion.py` | The checker: run it on a new recording |
| `source/` | The robot's own files, taken from the robot on 2026-10-25: the URDF, the MoveIt limits, the servo settings, the collision rules (SRDF) and the collision meshes |
| `libs/` | The collision library the checker uses, for Linux, Python 3.12 |
| `build_limits.py` | Rebuilds the YAML from `source/` (you should not need it) |

## The rules

| Rule | Limit | Notes |
|---|---|---|
| **Joint positions** | Inside every joint's limits in `tomo_re_uhand_limits.yaml`; **keep 2° away** from each limit (`safe_lower_rad` / `safe_upper_rad`) | The controller freezes the whole arm when a joint gets within 0.57° of its limit, and refuses any motion past it |
| **Speed, every joint** (arms, head **and fingers**) | Aim for **≤ 34°/s** (0.6 rad/s). **Never above 52°/s** (0.9 rad/s) | 34°/s is the robot's normal speed; 52°/s is its hard cap. Anything faster is slowed down on the robot, so it no longer matches the animation |
| **Acceleration** | ≤ 57°/s² (1.0 rad/s²) | Soft: the robot smooths it. Avoid sudden starts and stops |
| **No touching** | No two parts of the robot may touch: fingers of one hand, one hand against the other, a hand against the table | Only the pairs listed as allowed in `source/tomo_config.srdf` (mostly neighbouring links) may touch. Every finger of a hand is checked against every other finger |
| **Keep a gap** | When parts come close, keep **at least 10 mm** between them | Neighbouring fingers that are always about 5 mm apart (by design) are fine. The robot is never exactly on the animation; a few degrees of error closed a 6.4 mm gap on 2026-10-25 and stopped the run |

## Change your robot file first

Your rig's robot file (exported as `full_sequnece_00X.urdf`) has **two wrong shoulder limits**. That is why the animation could go past the real limits:

| Joint | Your file | **Real robot** |
|---|---|---|
| `Left_arm_joint2` | −16° … 169° | **−14° … 164°** |
| `Right_arm_joint2` | −166° … 19° | **−164° … 14°** |

All other joints already match. Use `source/robot_description.urdf` (the robot's own), or change these two limits in your rig.

## What to fix in the animation

Fix these in `iPhone_Packing_recording_ready.blend`. Frame numbers are from `full_sequnece_001` (30 fps, frames 0–1115); the other recordings are the same.

1. **Shoulders past their limit: frames 9–38 (READY).** `Left_arm_joint2` goes to −16° and `Right_arm_joint2` to +19°; the limits are ±14°. Keep them within **±12°**.
2. **Hands touching: frames 918–995 (PACK_IN_HOLDER).**
   - The left thumb (`LF_link_12`, `LF_link_13`) touches the right hand.
   - At frame 1031 the right thumb (`RF_link_13`) touches its own palm.
3. **Thumb too close to the index finger: frames 254–363 (PLACE_PHONE).** The left thumb tip is only 6.4 mm from the index finger. **This is the move that stopped the real robot.**
   - The thumb swings out to 70°, the fingers curl, then the thumb swings back across them.
   - Keep at least 10 mm, for example by bringing the thumb back before the fingers curl.
4. **Fingertips almost on the table: frames 673–685 (FLIP_BOX).** Both hands' fingertips are 2.3 mm above the table. Lift them to at least 10 mm.
5. **Too fast: 38 joints, the whole job.**
   - Examples: fingers up to 944°/s (frame 730), a wrist 584°/s (frame 655), against the 52°/s cap.
   - At the robot's cap this motion takes at least about 77 s; the animation does it in 37 s.
   - Slow the animation down (retime it) so no joint goes above 34–52°/s.
   - When you slow it down, keep the step markers (`Pick_iPhone`, `Release_iPhone`, …) and the carton, phone and tape animations in step with the new timing.

`report_full_sequnece_001.txt` lists every case. The `.json` has all of them, including the smaller near-misses.

## Check a new recording before generating data

```
python3 check_motion.py <your recording>.mmrt.yaml
```

- **Needs:** Python with `numpy` and `pyyaml`. The collision library is bundled in `libs/` for Linux + Python 3.12. On anything else run `pip install trimesh python-fcl` first.
- **Takes:** about 1.5 min per recording. `--clearance-mm 0` skips the near-miss part and takes a few seconds.
- **Result:** it ends with **`OK for the robot`** (exit code 0) or lists what is wrong with frame numbers.
- **Step names in the output:** add `--segments segments.json` from any generated episode (same timeline).
- **Also works on** a generated episode folder or a whole dataset folder.

Please only generate the dataset from a recording that passes.

## Where the numbers come from

Read from the robot itself on 2026-10-25 (the TOMO re_uhand configuration on the robot computer):

- joint limits: the controller's URDF;
- per-joint speed and acceleration limits: MoveIt `joint_limits.yaml`;
- the 0.01 rad (0.57°) freeze margin: `servo_config.yaml`;
- which link pairs may touch: `tomo_config.srdf`;
- the 0.6 / 0.9 rad/s speeds and 1.0 rad/s² acceleration: the robot brain's `robot.yaml`.

The checker was tested against the real robot's own collision check: it flags exactly the finger pair that stopped the robot on 2026-10-25, and it flags nothing in the poses the robot actually passed through.
