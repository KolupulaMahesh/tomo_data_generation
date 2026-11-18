# TOMO full-sequence dataset generator (Isaac Sim only)

Generates the full iPhone-packing sequence (pick phone → place in carton → close → tape → flip → holder → box out → home,
1116 frames / 37 s at 30 fps) with a **different iPhone start pose per episode**: the right arm is re-solved by IK to reach
it, the phone is carried by the new hand, everything after the carry plays back as recorded. Also random phone box
(iPhone 14 Plus / 14 Pro Max print, random hue) and random table colour.

## Requirements
- **Isaac Sim 6.0** (tested with 6.0.0-rc.59), NVIDIA RTX GPU. Default path `~/isaac-sim/python.sh` (override with `ISAAC=...`).
- A Python with `numpy` and `opencv-python` for packing (`PY=...`, default `python3`).
- Disk: ~60–90 MB per episode (+ ~3 GB temporary per episode while rendering).

## Run
```bash
cd finalfullsequncepraveen
N=5 bash tools/run_isaac_batch.sh                       # test 5 episodes first
N=250 bash tools/run_isaac_batch.sh                     # full dataset (resumable: finished episodes are skipped)
# custom paths:
ISAAC=/opt/isaac-sim/python.sh PY=/usr/bin/python3 N=250 bash tools/run_isaac_batch.sh
# long run in the background (Linux):
N=250 setsid nohup bash tools/run_isaac_batch.sh > batch.log 2>&1 < /dev/null &
tail -f batch.log
```
Episode 0 keeps the recorded phone pose; the others are random (seeded: `SEED=0` default -> reproducible).
Several machines can split the work with different `SEED` values and output folders (`OUT=...`).

## Output (`dataset_isaac/fs_XXX/`)
| file | content |
|---|---|
| `rgb_jpg.npy` + `rgb_off.npy` | RGB frames, 960x540 JPEG (q95), concatenated bytes + offsets |
| `depth_png.npy` + `depth_off.npy` | depth, 960x540 16-bit PNG in **mm**, 0 = no reading |
| `q.npy` | 46 joint targets per frame (order in `joint_names.json`) |
| `suction.npy` | right-hand suction 0/1 |
| `phone_world.npy`, `box_world.npy` | 4x4 world poses of the iPhone box and the carton |
| `phone_uv.npy`, `phone_visible.npy` | phone centre in camera pixels (1920x1080) + visible flag |
| `labels.npz` | phase one-hot (9 phases), task_progress, done, carton bones, … |
| `segments.json` | phase start/end frames, release frame |
| `variant.json` | settings of this episode (phone offset, box, table, camera K, IK accuracy) |
| `preview.mp4` | video of the episode |

## Camera (Tomo head camera, as on the robot)
Render 1920x1080, fx 1070.789795, fy 1070.526978, cx 962.309387, cy 550.0, no distortion, pose = URDF Head_camera_link.
Saved: whole frame at 960x540 (colour `cv2.INTER_AREA`, depth `cv2.INTER_NEAREST`, no reading = 0).
`--cam original` instead gives the 4:3 view of the original recording video.

## Single episode / options
```bash
<isaac>/python.sh tools/fs_isaac_gen.py --out dataset_isaac/my_ep --dx 0.05 --dy -0.03 --dyaw 15 \
    --table-rgb 0.75,0.72,0.66 --phone-tex assets/phone_textures/iphone_14_pro_max_texture_0.2x.png --phone-hue 40
python3 tools/fs_pack.py dataset_isaac/my_ep
```
`--dx/--dy` metres, `--dyaw` degrees (reachable range tested: within 10 cm, ±25°), `--stride 2` = 15 fps.

## Folder
- `tools/` – `fs_isaac_gen.py` (scene, IK, render), `fs_pack.py` (episode format), `run_isaac_batch.sh` (batch)
- `isaac_base/` – the recorded scene + motion (exported once from the original Blender file)
- `robot_scene/` – TOMO robot USD
- `assets/` – room backdrop photo, iPhone 14 Pro Max box print
