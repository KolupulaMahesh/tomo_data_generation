"""The rig's own motion (rig.py: curves + constraints + Blender's IK, nothing taken from Blender's result), for Isaac.

Generic (2026-10-29, render_episode.py's input), for any scene whose look was exported (blender_scripts/
export_scene_usd.py): <run>/generic/motion.npz - per frame the world matrix of every exported object that moves, the
armature-space pose of every bone of the skeletons outside the robot, the robot's 46 joints, the robot's IO byte from
the .blend's markers (io0..io7 = bits 0..7) and the labels scenes/<scene>.yaml asks for - and <run>/generic/episode.json
(scenes/<scene>.yaml: what the .blend does not say, + this episode). The 46 joints are each bone's rotation about its axis
relative to its rest pose (axes found against the dataset's stored motion: arms / head Z; finger base joints X or Y, as
the MMRT recorder reads them).

For the packing scene's original and hand-moved runs also the frozen generator's motion file (isaac_base/
base_motion.npz: the 46 joints + the phone pose; every other key copied from the original), in <run>/isaac_pkg/ (a mirror
of renderer/ whose motion file is ours), so render_in_isaac.sh renders the motion THIS rig computed.

    python make_motion.py [scene]          -> runs/<scene>/
    python make_motion.py scene DX DY DYAW -> the phone moved first by the Synthesizer (synth.py, the data team's setup,
                                              checked against Blender's own) -> runs/<scene>/synth_<tag>/
    python make_motion.py scene --seed N   -> dataset episode N: every object the .blend's Synthesizer moves gets a
                                              random offset inside the range stored in the .blend, plus a random look
                                              (scenes/<scene>.yaml random_look); the same N gives the same episode
                                              -> runs/<scene>/ep_<N>/  [--bars: the live camera's black bars painted in]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rig import Rig  # noqa: E402
from scene_robot import recorder_steps, robot_setup  # noqa: E402
from synth import PHONE_SETUP, Synthesizer, blend_setup, phone_offsets, random_offsets  # noqa: E402

# ---------------- settings ----------------
HERE = Path(__file__).resolve().parent
EXPORTS = HERE / "blender_export"                                  # blender_scripts/export_rig.py's output (frozen copy)
GEN = HERE / "renderer"                                            # the dataset's Isaac generator (frozen copy)
SCENE = sys.argv[1] if len(sys.argv) > 1 else "iPhone_Packing_Gidora_6_synth_8"
ARGS = sys.argv[2:]
SEED = int(ARGS[ARGS.index("--seed") + 1]) if "--seed" in ARGS else None
OFFSET = [float(x) for x in ARGS[:3]] if SEED is None and len(ARGS) >= 3 else None    # phone dx, dy (m), dyaw (deg)
RUN = HERE / "runs" / SCENE / (f"ep_{SEED:04d}" if SEED is not None else
                               "synth_" + "_".join(f"{v:+g}" for v in OFFSET) if OFFSET else "")
LOOK = EXPORTS / SCENE / "scene"                                   # blender_scripts/export_scene_usd.py's output
CFG = HERE / "scenes" / f"{SCENE}.yaml"
FROZEN_SCENE = "iPhone_Packing_local"                              # the only scene the frozen generator was made for
SETTINGS = yaml.safe_load(CFG.read_text()) if CFG.exists() else {}
ROBOT_SET = robot_setup(SETTINGS, CFG.parent)                      # scene_robot.py: which robot this scene has
ROBOT = ROBOT_SET["object"]                                        # the Blender robot's root object (drawn "usd": the renderer
                                                                   # puts the robot's own USD in its place)
ROBOT_JOINTS = ROBOT_SET["joints"]                                 # the dataset's joint order


def armature_of(joint):
    return ROBOT_SET["armatures"].get(joint[:2], ROBOT)


def joint_value(rig, joint):
    arm = armature_of(joint)
    bd = rig.bone[(arm, joint)]
    base = rig.pose(arm, bd["parent"]) @ bd["offs"] if bd["parent"] else bd["ml"]
    R = np.linalg.inv(base[:3, :3]) @ rig.pose(arm, joint)[:3, :3]
    i, j = [(1, 2), (2, 0), (0, 1)][ROBOT_SET["axes"].get(joint, 2)]
    return float(np.arctan2(R[j, i], R[i, i]))


def relay_per_frame(rig, frames):
    """The robot's IO byte per frame from the .blend's markers (MMRT IO_CONTROL events: io0..io7 -> bits 0..7, each
    event holding until the next)."""
    r = np.zeros(len(frames), np.uint8)
    ev = sorted((m["frame"], sum(int(e[f"io{k}"]) << k for k in range(8)))
                for m in rig.doc["scene_mmrt"].get("mmrt_markers", []) for e in m["events"]
                if e["flag_type"] == "IO_CONTROL")
    for f, v in ev:
        r[frames >= f] = v
    return r


def label_values(rig, spec):
    """scenes/<scene>.yaml labels at the current frame: {world: object} -> 4x4, {influence: [owner, constraint]} -> 0..1."""
    return {n: rig.world(s["world"]) if "world" in s else float(rig.props(s["influence"][0])["influence"][s["influence"][1]])
            for n, s in spec.items()}


def random_look(cfg, rng, seed=None):
    """This episode's colours and prints, from scenes/<scene>.yaml random_look.
    balanced: true -> no dice: the seed picks the next table colour (seed % colours) and the next print
    ((seed // colours) % prints), so every colours x prints seeds hold each combination exactly once."""
    rl = cfg.get("random_look", {})
    if rl.get("balanced") and seed is not None:
        recolor, retexture, nc = {}, {}, 1
        for n, r in rl.get("recolor", {}).items():
            nc = len(r["colours"])
            c = r["colours"][seed % nc]
            if c is not None:                                       # None = the .blend's own colour
                recolor[n] = [float(v) for v in c]
        for n, r in rl.get("retexture", {}).items():
            retexture[n] = {"image": r["images"][(seed // nc) % len(r["images"])], "hue": 0.0}
        return {"recolor": recolor, "retexture": retexture}
    recolor = {n: [round(float(v), 3) for v in rng.uniform(r["low"], r["high"])]
               for n, r in rl.get("recolor", {}).items() if rng.uniform() >= r.get("keep", 0.0)}
    retexture = {n: {"image": r["images"][int(rng.integers(len(r["images"])))],
                     "hue": round(float(rng.uniform(*r["hue"])), 1)} for n, r in rl.get("retexture", {}).items()}
    return {"recolor": recolor, "retexture": retexture}


def episode_settings(cfg, out, offsets, look, markers, doc):
    """scenes/<scene>.yaml + this episode -> episode.json (image paths relative to the episode folder: it can move)."""
    rel = lambda p: os.path.relpath(HERE / p, out) if p else None   # noqa: E731
    return {"scene": SCENE, "seed": SEED, "phone_offset_dx_dy_dyaw": OFFSET, "markers": markers,
            "offsets": {k: [list(t), list(r)] for k, (t, r) in offsets.items()} if offsets else None,
            "look_dir": os.path.relpath(LOOK, out), "motion": "motion.npz", "robot": ROBOT,
            "hide": cfg.get("hide", []), "hide_from_frame": cfg.get("hide_from_frame", {}),
            "light_gain": cfg.get("light_gain", 1.0), "dome": cfg.get("dome", 0.0),
            "recolor": look.get("recolor", {}),
            "retexture": {k: {**v, "image": rel(v.get("image"))} for k, v in look.get("retexture", {}).items()},
            "steps": cfg.get("steps") or recorder_steps(doc), "vacuum_io": cfg.get("vacuum_io", {}), "track": cfg.get("track"),
            "camera_bars": cfg.get("live_camera_bars") if cfg.get("camera_bars") or "--bars" in ARGS else None,
            **({"robot_drawn": ROBOT_SET["drawn"]} if ROBOT_SET["drawn"] != "usd" else {}),
            **({"camera": cfg["camera"]} if cfg.get("camera") else {})}


def write_frozen(q, phone, T, t0):
    """The frozen generator's package: renderer/ mirrored into <run>/isaac_pkg with our motion file."""
    orig = np.load(GEN / "isaac_base/base_motion.npz")
    assert [str(n) for n in orig["names"]] == ROBOT_JOINTS and len(orig["q"]) == T, "not the frozen generator's scene"
    out = RUN / "isaac_pkg"
    (out / "isaac_base").mkdir(parents=True, exist_ok=True)
    (out / "tools").mkdir(exist_ok=True)
    for src, dst in [(GEN / "isaac_base/base.usdc", out / "isaac_base/base.usdc"),
                     (GEN / "isaac_base/textures", out / "isaac_base/textures"),
                     (GEN / "robot_scene", out / "robot_scene"), (GEN / "assets", out / "assets"),
                     (GEN / "tools/fs_isaac_gen.py", out / "tools/fs_isaac_gen.py"),
                     (GEN / "tools/fs_pack.py", out / "tools/fs_pack.py"),
                     (GEN / "tools/joint_names.json", out / "tools/joint_names.json")]:
        if dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.relpath(src, dst.parent), dst)        # the generator unchanged; its root = this folder
                                                                 # (relative: the whole folder can move)
    data = {k: orig[k] for k in orig.files}
    data.update(q=q.astype(np.float32), phone_world=phone)
    np.savez(out / "isaac_base/base_motion.npz", **data)
    dq = np.degrees(np.abs(np.angle(np.exp(1j * (q.astype(np.float32).astype(np.float64) - orig["q"])))))
    rep = {"scene": SCENE, "phone_offset_dx_dy_dyaw": OFFSET, "frames": T, "seconds": round(time.time() - t0, 1),
           "max_joint_diff_deg_vs_stored": round(float(dq.max()), 4),
           "worst_joint": ROBOT_JOINTS[int(dq.max(0).argmax())],
           "max_phone_diff_vs_stored": float(np.abs(phone - orig["phone_world"]).max())}
    (out.parent / "motion_check.json").write_text(json.dumps(rep, indent=1))
    print(f"[motion] frozen generator's motion file -> {out / 'isaac_base/base_motion.npz'}")
    print(f"[motion] vs the dataset's stored motion: joints max {rep['max_joint_diff_deg_vs_stored']} deg "
          f"({rep['worst_joint']}), phone max {rep['max_phone_diff_vs_stored']:.1e}")


def main():
    t0 = time.time()
    if not (LOOK / "export.json").exists():
        sys.exit(f"no look export for {SCENE}: run blender_scripts/export_scene_usd.py first ({LOOK})")
    look = json.loads((LOOK / "export.json").read_text())
    if look.get("unsupported_animation"):
        sys.exit(f"{SCENE}: Blender animates things the rig does not read: {look['unsupported_animation'][:5]}")
    cfg = SETTINGS
    rig = Rig(EXPORTS / SCENE / "rig.json")
    offsets, ep_look = None, cfg.get("look", {})
    if SEED is not None:
        rng = np.random.default_rng(SEED)
        offsets = random_offsets(rig, rng)
        ep_look = random_look(cfg, rng, SEED)
        Synthesizer(rig, blend_setup(rig)).run(offsets)
    elif OFFSET:
        offsets = phone_offsets(*OFFSET)
        Synthesizer(rig, PHONE_SETUP).run(offsets)
    frames = np.arange(rig.doc["frame_start"], rig.doc["frame_end"] + 1)
    T = len(frames)
    robot = look["objects"][ROBOT]
    inside = lambda p: ROBOT_SET["drawn"] == "usd" and (p == robot or p.startswith(robot + "/"))   # noqa: E731  (its USD replaces these)
    objs = [n for n, p in look["objects"].items() if not inside(p)]
    bones = [(s["armature"], b) for s in look["skeletons"] if not inside(s["prim"]) for b in s["bones"]]
    spec = cfg.get("labels", {})
    q = np.zeros((T, len(ROBOT_JOINTS)))
    W, P, R = np.zeros((T, len(objs), 4, 4)), np.zeros((T, len(bones), 4, 4)), np.zeros((T, 4, 4))
    labels = {n: [] for n in spec}
    for i, f in enumerate(frames):
        rig.evaluate(float(f))                                   # nothing injected: all ours
        q[i] = [joint_value(rig, n) for n in ROBOT_JOINTS]
        W[i] = [rig.world(n) for n in objs]
        P[i] = [rig.pose(a, b) for a, b in bones]
        R[i] = rig.world(ROBOT)
        for n, v in label_values(rig, spec).items():
            labels[n].append(v)
    moving = np.flatnonzero(np.abs(W - W[:1]).max(axis=(0, 2, 3)) > 0)
    out = RUN / "generic"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "motion.npz", fps=rig.doc["fps"] / rig.doc["fps_base"], frames=frames,
             robot_joints=np.array(ROBOT_JOINTS), robot_q=q, robot_world=R,
             names=np.array(ROBOT_JOINTS), q=q,                     # (robot_limits/check_motion.py reads these two)
             object_names=np.array([objs[k] for k in moving]), object_world=W[:, moving],
             world0_names=np.array(objs), world0=W[0],              # every object at frame 0, randomised ones too (objects_file.py)
             bone_owner=np.array([a for a, _ in bones]), bone_names=np.array([b for _, b in bones]), bone_pose=P,
             relay=relay_per_frame(rig, frames), label_names=np.array(list(spec)),
             **{f"label_{n}": np.array(v) for n, v in labels.items()})
    markers = sorted(([m.get("marker_name") or m["name"], m["frame"]]          # MMRT's name, without the uid
                      for m in rig.doc["scene_mmrt"].get("mmrt_markers") or rig.doc["markers"]), key=lambda x: x[1])
    (out / "episode.json").write_text(json.dumps(episode_settings(cfg, out, offsets, ep_look, markers, rig.doc), indent=1))
    print(f"[motion] {SCENE}{f' episode {SEED}' if SEED is not None else ''}: {T} frames by the rig in "
          f"{time.time() - t0:.1f} s -> {out}: {len(moving)} of {len(objs)} objects move, {len(bones)} bones, "
          f"{len(ROBOT_JOINTS)} robot joints" + (f" | offsets {offsets}" if SEED is not None else ""))
    if SCENE == FROZEN_SCENE and SEED is None:
        write_frozen(q, W[:, objs.index("iPhone")], T, t0)


if __name__ == "__main__":
    main()
