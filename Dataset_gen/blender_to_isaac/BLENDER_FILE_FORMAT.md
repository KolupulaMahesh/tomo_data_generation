# Blender file format for the dataset generator

What a Blender scene must follow so the dataset generator (`Dataset_gen/blender_to_isaac`) can turn it into a training
dataset: random object positions, the robot's motion re-solved, rendered in Isaac Sim, packed for training.

These rules come from the generator's code (2026-10-29). Anything in **Must not have** stops the generator with an
error naming the cause, so a wrong file never turns silently into wrong data.

The packing scene `vla_full_sequence/blender_work/iPhone_Packing_local.blend` follows every rule and is the reference
example throughout.

---

## Quick checklist

- [ ] The TOMO robot rig, unchanged (`tomo` + both hands, 46 joints)
- [ ] The whole job in one scene, frame range and fps set
- [ ] Motion made only from the supported building blocks (keys, 5 constraint types, standard IK)
- [ ] MMRT Synthesizer panel filled in for the object(s) to randomise
- [ ] MMRT markers with IO events at every vacuum switch
- [ ] No drivers, NLA strips, animated materials/lights/cameras, or meshes that change shape
- [ ] Materials: Principled BSDF with image textures
- [ ] Linked `.blend` files present at their paths

---

## 1. Must have

### 1.1 The TOMO robot rig, unchanged

| What | Name in the file |
|---|---|
| Robot root armature | `tomo` |
| Hand armatures | `tomo_right_hand`, `tomo_left_hand` |
| Head joints | `Head_joint1`, `Head_joint2` |
| Arm joints | `Left_arm_joint1` … `Left_arm_joint7`, `Right_arm_joint1` … `Right_arm_joint7` |
| Finger joints | `RF_joint_11` … `RF_joint_53`, `LF_joint_11` … `LF_joint_53` (5 fingers × 3 joints per hand) |

- Link it from `tomo_re.blend` and `uHand_thick.blend` (`vla_full_sequence/vla_full_sequence/blender_files/tomo_common/`),
  as the packing file does. Do not rename or re-rig it.
- The robot can stand anywhere in the scene: the generator puts the robot's own model where `tomo` stands.

### 1.2 The whole job in one scene

- One scene holds the whole task, with **frame start / end** and **fps** set (packing: frames 0–1115 at 30 fps).
- Motion = keyframes on:
  - objects: location, rotation, scale;
  - pose bones: location, rotation, scale;
  - constraint influence, e.g. the object's `Child Of pick` switched on at the grasp and off at the release.

### 1.3 Only these motion building blocks

| Kind | Supported |
|---|---|
| Object / bone constraints | Child Of (all 9 channels on), Damped Track, Transform, Limit Rotation, Stretch To |
| Inverse kinematics | Blender's **standard IK** solver (not iTaSC); each IK chain uses its own bones |
| Rotation mode | Euler (any order) or Quaternion — not Axis-Angle |
| Parenting | to an object, a bone, or an armature (armature deform) — not to a curve or vertices |
| Constraint spaces | objects: World, Local; bones: World, Pose, Local, Local With Parent |

### 1.4 MMRT Synthesizer panel filled in

The generator reads the random ranges from the file's own Synthesizer settings. For each object to randomise:

| Field | Meaning | Packing example (`iPhone`) |
|---|---|---|
| Object | the object that moves to a random place | `iPhone` |
| Frame min / max | the frames over which it is moved | 0 – 155 |
| Translation min / max | x, y, z range in metres | x −0.15 … +0.10, y −0.04 … +0.10, z 0 |
| Rotation min / max | roll, pitch, yaw range in degrees | yaw −30 … +30 |
| Parent offset object | follow another randomised object instead of a range | — |

The arm's IK control must follow the randomised object over the grasp. In the packing file:
`Right_arm_joint7_ik_control.001`, frames 127–165, **parent offset object `iPhone`**, ranges all zero.

> Without a Synthesizer setup every episode is the same motion; only the look changes. The generator window says so.

### 1.5 MMRT markers with IO events

- Put a marker with an **IO_CONTROL** event at every vacuum switch. Each event sets the robot's IO pins io0 … io7;
  the setting holds until the next IO event.
- **io0 = right vacuum, io2 = left vacuum** (the same robot in every scene).
- The marker names (MMRT's name, without the id) become the dataset's step names, unless the scene's settings file
  lists its own steps.

Packing example:

| Frame | Marker | IO |
|---|---|---|
| 155 | Pick_iPhone | io0 on (right vacuum) |
| 290 | Release_iPhone | all off |
| 492 | Pick_Ripcord | io2 on (left vacuum) |
| 594 | Align_Ripcord | io0 + io2 on |
| 608 | Paste_Ripcord | all off |
| 1060 | Pick_Box | io0 + io1 on |
| 1095 | Drop_Box | all off |

---

## 2. Must not have (the generator stops and names the cause)

| Not allowed | Why | Where it is caught |
|---|---|---|
| Drivers, NLA strips | the generator reads each object's own keyframes only | scene export → `export.json` `unsupported_animation` |
| Animated materials, lights, cameras, world, shape keys | not rebuilt in Isaac | same |
| Meshes that change shape over time other than through bones (cloth, shape keys, curve / lattice deform, animated geometry nodes) | Isaac shows the shape of frame 0 only | scene export → `geometry_changes` (measured at 5 frames) |
| Two objects with the same name (e.g. one linked from a library) | objects are matched by name | scene export |
| Any other constraint (Copy Location, Copy Rotation, Track To, Follow Path, Armature, …) | not rebuilt yet (each one can be added, ~30–60 min, checked against Blender) | motion step |
| iTaSC IK, Axis-Angle rotation, a curve or vertex parent | not rebuilt | motion step |

Fixed modifiers that do not change over time are fine: Subdivision, Smooth by Angle, Mirror, Solidify, Bevel.

---

## 3. Should have (otherwise the pictures may look different)

- **Materials:** Principled BSDF with image textures. An image plugged straight into the material's Surface (or
  through an Emission shader) is also fine: it is shown self-lit, like the room photo in the packing scene. Mix shaders
  and procedural textures do not carry over properly — check the first render.
- **Hidden objects:** anything that must not appear in the pictures is hidden from render in Blender (object or
  collection), or listed under `hide` in the scene's settings file. Objects with "Visibility → Camera" off are hidden
  in Isaac too.
- **Linked files:** every linked `.blend` (props, jigs, robot) is at its path when the scene is added.
- **Units:** metres, Z up (Blender's defaults).
- **Lighting:** ordinary Blender lights. They are converted with the packing scene's calibration (`light_gain: 300`
  plus a dome light of 2500); a very different lighting setup may need `light_gain` changed.

---

## 4. Not needed in the .blend

- **A camera:** the robot's head camera with the live lens (1920×1080, fx 1070.79, fy 1070.53, cx 962.31, cy 550.0) is
  always used.
- **Random looks** (colours, prints): set in the scene's settings file.

---

## 5. Done by hand for each new scene (about 10 minutes)

### 5.1 The settings file `scenes/<name>.yaml`

Made from a template when the scene is added; fill in what the .blend does not say:

| Field | What to put | Packing example |
|---|---|---|
| `hide` | objects to leave out of the pictures | `[Plane.002, Plane.009]` (floor and ceiling) |
| `hide_from_frame` | `{object: frame}`: not drawn from that frame on | `{iPhone: 445}` (lid shut) |
| `light_gain`, `dome` | light calibration | `300`, `2500` |
| `track` | the object labelled as the target (`phone_world` / `phone_uv` / `phone_visible`) | `iPhone` |
| `steps` | `[[NAME, first frame], …]`; leave out to use the marker names | the 9 packing steps |
| `vacuum_io` | IO pins of the two vacuums | `{right: 0, left: 2}` |
| `labels` | extra per-frame labels: `{world: object}` or `{influence: [object, constraint]}` | `box_world`, `inf_pick`, … |
| `random_look` | colours / prints varied per episode | table colour, phone print + hue |
| `camera_bars` | keep `false` | `false` |
| `task_yaml`, `task_files` | the training's task file and its extra files | `scenes/iPhone_Packing_local.task.yaml`, `carton_hinges.npz` |

### 5.2 The training's task file `scenes/<name>.task.yaml`

Needed only to **train** on the dataset (the dataset itself can be made without it): the step names with their
sentences, the substeps, which label is which vacuum. Copy `scenes/iPhone_Packing_local.task.yaml` and change the step
names and sentences. When a dataset stops early (e.g. "up to the end of CLOSE_CARTON"), the generator cuts this file to
the steps the episodes contain.

---

## 6. Adding and checking a new scene

**In the window** — `bash dataset_generator.sh`:

1. **Add a Blender file…** → choose the `.blend` → give a short name. Blender only reads the file; the two exports take
   about 30 seconds.
2. Read the scene information the window shows: frames, the random ranges found, the steps, and any problem.
3. **Open its settings file** and fill in section 5.1.
4. Generate **2 episodes** and watch `datasets/<name>_<date>/fs_*/preview_h264.mp4`.

**Or in a terminal**, from `Dataset_gen/blender_to_isaac`:

```bash
blender -b <file>.blend --python blender_scripts/export_rig.py -- blender_export/<name>
blender -b <file>.blend --python blender_scripts/export_scene_usd.py -- blender_export/<name>
# write scenes/<name>.yaml (copy the template from dataset_generator.py or the packing scene's file)
bash generate.sh <name> 2
```

**Proof that the motion is Blender's** (recommended for each new file): the rig export also writes Blender's own
result for every frame, and these compare the generator's motion with it:

```bash
env/bin/python check_part2.py <name>
env/bin/python check_part3.py <name>
```

---

## 7. What every episode gets

- The randomised object(s) at a random place inside the Synthesizer ranges; the same episode number always gives the
  same episode.
- The robot's motion re-solved by the rig (Blender's own IK and constraints).
- A robot-limits report (`limits.json`: joint limits, speed, self-collision).
- Colour + depth from the robot's head camera, saved at 960×540, in the format the training reads (`fs_NNN` folders,
  `labels.npz` with `suction`, `left_suction` and the IO byte `relay`, `segments.json`, `preview_h264.mp4`).
