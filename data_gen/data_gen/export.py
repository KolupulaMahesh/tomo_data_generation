"""Writing an episode.

Always (every source):
  recovery_labels.npz   per frame: demo frame, stage, phase, failure type, recovery strategy, attempt, grip IO, joints.
                        Things the robot can know or the training target: safe to train on.
  sim_truth.npz         per frame: every object's exact world pose, held by which hand. Simulation-only (privileged):
                        for checking and evaluation, NOT a model input.
  events.json           the failure / detection / decision / verification / resume events in order
  validation.json       validate.py's result
  preview.mp4           (optional) a top-view sketch of hands and objects with the phase written on it

With a blender_to_isaac motion.npz source (for Isaac Sim): motion.npz + episode.json in render_episode.py's format.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

B2I = Path(__file__).resolve().parents[2] / "Dataset_gen/blender_to_isaac"


def _codes(values):
    names = sorted(set(values), key=lambda v: (v != "", v))
    m = {v: k for k, v in enumerate(names)}
    return np.array([m[v] for v in values], np.int16), np.array(names)


def write_labels(b, out: Path, meta, validation):
    out.mkdir(parents=True, exist_ok=True)
    o = b.out
    from .builder import PHASES
    phase = np.array([PHASES.index(p) for p in o["phase"]], np.int8)
    stage_names = [s for s, _ in b.task["stages"]]
    stage = np.array([stage_names.index(s) for s in o["stage"]], np.int16)
    fail, fail_names = _codes(o["failure"])
    rec, rec_names = _codes(o["recovery"])
    relay = np.array(o["relay"], np.uint8)
    grip = {f"grip_{h}": ((relay & b.hand_bits(h)) > 0).astype(np.uint8) for h in b.hands}
    np.savez_compressed(out / "recovery_labels.npz",
                        demo_frame=np.array(o["src"], np.int32), q=np.array(o["q"], np.float32),
                        joint_names=np.array(b.s.joints), relay=relay, **grip,
                        stage=stage, stage_names=np.array(stage_names),
                        phase=phase, phase_names=np.array(PHASES),
                        failure=fail, failure_names=fail_names, recovery=rec, recovery_names=rec_names,
                        attempt=np.array(o["attempt"], np.int8), fps=b.s.fps)
    names = list(b.obj_out)
    np.savez_compressed(out / "sim_truth.npz", object_names=np.array(names),
                        object_world=np.stack([np.array(b.obj_out[n]) for n in names], 1).astype(np.float32),
                        note="simulation-only ground truth: not a model input")
    (out / "events.json").write_text(json.dumps({"meta": meta, "events": b.events}, indent=1, default=_js))
    (out / "validation.json").write_text(json.dumps(validation, indent=1, default=_js))


def _js(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# --------------------------------------------------------------------------------------------- preview sketch
def preview(b, path: Path, task, every=1):
    import cv2
    W, H, s = 960, 720, 520.0                                 # px per metre, robot base near the bottom middle
    to_px = lambda p: (int(W / 2 - p[1] * s), int(H - 90 - p[0] * s))   # noqa: E731  (x forward = up, y left = left)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), b.s.fps / every, (W, H))
    (x0, x1), (y0, y1) = task["table_xy"]
    col = {"normal": (90, 200, 90), "failure": (40, 40, 230), "detect": (0, 170, 255), "recovery": (230, 160, 40),
           "verify": (230, 230, 40), "resume": (200, 120, 200), "terminal": (0, 0, 160)}
    q = b.out["q"]
    for i in range(0, b.n, every):
        img = np.full((H, W, 3), 30, np.uint8)
        cv2.rectangle(img, to_px([x1, y0]), to_px([x0, y1]), (70, 70, 70), -1)
        base = b.s.robot_world[b.out["src"][i]]
        for h, hc in b.hands.items():
            pts = [b.kin.link(n, q[i], base)[:3, 3] for n in (hc["arm"][0].replace("joint1", "link1"), hc["link"])]
            p = to_px(pts[1])
            cv2.line(img, to_px(pts[0]), p, (160, 160, 160), 3)
            on = b.out["relay"][i] & b.hand_bits(h)
            cv2.circle(img, p, 9, (0, 255, 255) if on else (200, 200, 200), -1)
            z = pts[1][2]
            cv2.putText(img, f"{h[0].upper()} z{z:.2f}", (p[0] + 10, p[1]), 0, 0.45, (220, 220, 220), 1)
        for n, L in b.obj_out.items():
            P = L[i]
            p = to_px(P[:3, 3])
            tip = to_px(P[:3, 3] + P[:3, 0] * 0.06)
            cv2.circle(img, p, 6, (255, 120, 60), -1)
            cv2.line(img, p, tip, (255, 120, 60), 2)
            cv2.putText(img, f"{n} z{P[2, 3]:.3f}", (p[0] + 8, p[1] - 8), 0, 0.45, (255, 170, 120), 1)
        ph = b.out["phase"][i]
        cv2.rectangle(img, (0, 0), (W, 64), col.get(ph, (90, 90, 90)), -1)
        txt = f"{i:4d} demo {b.out['src'][i]:4d} | {b.out['stage'][i]} | {ph}"
        cv2.putText(img, txt, (10, 26), 0, 0.7, (0, 0, 0), 2)
        sub = " ".join(x for x in (b.out["failure"][i], b.out["recovery"][i],
                                   f"attempt {b.out['attempt'][i]}" if b.out["attempt"][i] else "") if x)
        cv2.putText(img, sub, (10, 54), 0, 0.6, (0, 0, 0), 2)
        vw.write(img)
    vw.release()


# --------------------------------------------------------------------------------------------- Isaac (render_episode.py)
class RobotSkin:
    """the robot's bone poses from its joints, through the rig's own rest data (for a robot drawn from the .blend)."""

    def __init__(self, rig, robot_set):
        self.rig, self.rs = rig, robot_set
        self.joints = robot_set["joints"]
        self.axes = robot_set["axes"]

    def arm_of(self, j):
        return self.rs["armatures"].get(j[:2], self.rs["object"])

    def poses(self, frame, q):
        rig = self.rig
        rig.evaluate(float(frame))
        jq = dict(zip(self.joints, q))
        memo = {}

        def pose(arm, b):
            if (arm, b) in memo:
                return memo[(arm, b)]
            bd = rig.bone[(arm, b)]
            base = pose(arm, bd["parent"]) @ bd["offs"] if bd["parent"] else bd["ml"]
            if b in jq:
                a = jq[b]
                c, s = np.cos(a), np.sin(a)
                ax = self.axes.get(b, 2)
                R = np.eye(4)
                i, j = [(1, 2), (2, 0), (0, 1)][ax]
                R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
                M = base @ R
            else:
                M = base @ rig.chan(arm, b)
            memo[(arm, b)] = M
            return M

        arms = {self.arm_of(j) for j in self.joints}
        return {(a, b): pose(a, b) for (a, b) in rig.bone if a in arms}


def write_isaac_episode(b, src_motion: Path, out: Path, scene: str):
    """motion.npz + episode.json for blender_to_isaac/render_episode.py (then pack_episode.py makes fs_NNN)."""
    sys.path.insert(0, str(B2I))
    import yaml
    from rig import Rig
    from scene_robot import robot_setup
    src_dir = Path(src_motion).parent
    M = np.load(src_motion)
    cfg = yaml.safe_load((B2I / "scenes" / f"{scene}.yaml").read_text())
    rs = robot_setup(cfg, B2I / "scenes")
    rig = Rig(B2I / "blender_export" / scene / "rig.json")
    skin = RobotSkin(rig, rs)
    src = np.array(b.out["src"])
    n = len(src)
    q = np.array(b.out["q"])
    owners, bnames = [str(x) for x in M["bone_owner"]], [str(x) for x in M["bone_names"]]
    BP = M["bone_pose"][src].copy()
    robot_arms = {skin.arm_of(j) for j in rs["joints"]}
    rob_cols = [k for k, a in enumerate(owners) if a in robot_arms]
    check = []
    for i in range(n):
        if rob_cols:
            P = skin.poses(src[i], q[i])
            for k in rob_cols:
                if (owners[k], bnames[k]) in P:
                    if np.allclose(q[i], M["robot_q"][src[i]]):
                        check.append(np.abs(P[(owners[k], bnames[k])] - BP[i, k]).max())
                    BP[i, k] = P[(owners[k], bnames[k])]
        for (arm, bone), Pv in b.bone_out[i].items():
            if arm not in robot_arms:
                k = next(k for k in range(len(owners)) if owners[k] == arm and bnames[k] == bone)
                BP[i, k] = Pv
    names = [str(x) for x in M["object_names"]]
    OW = M["object_world"][src].copy()
    on_robot = [k for k, nme in enumerate(names) if nme in b.s.robot_attached]
    for k, nme in enumerate(names):
        if nme in b.obj_out and nme not in b.s.robot_attached:
            OW[:, k] = np.array(b.obj_out[nme])
    rob_keys = [(owners[k], bnames[k]) for k in rob_cols]
    for i in range(n):                                    # robot link meshes, camera, ...: the rig's own hierarchy with the
        if np.allclose(q[i], M["robot_q"][src[i]]):       # robot's bones set from this frame's joints
            continue                                      # (the demo's joints: the demo's poses are already right)
        rig.evaluate(float(src[i]), inject={key: BP[i, c] for key, c in zip(rob_keys, rob_cols)})
        for k in on_robot:
            OW[i, k] = rig.world(names[k])
    rig.evaluate(float(src[0]), inject={key: BP[0, c] for key, c in zip(rob_keys, rob_cols)})
    check.append(max(np.abs(rig.world(names[k]) - M["object_world"][src[0], k]).max() for k in on_robot) if on_robot else 0)
    data = {k: M[k] for k in M.files}
    data.update(frames=np.arange(n), demo_frames=src.astype(np.int64), robot_q=q, q=q, robot_world=M["robot_world"][src], object_world=OW, bone_pose=BP,
                relay=np.array(b.out["relay"], np.uint8),
                **{k: M[k][src] for k in M.files if k.startswith("label_") and k != "label_names"})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "motion.npz", **data)
    ep = json.loads((src_dir / "episode.json").read_text())
    first = {}
    for i, f in enumerate(src):
        first.setdefault(int(f), i)
    remap = lambda f: first.get(int(f), int(np.searchsorted(src, f)))   # noqa: E731
    if ep.get("steps"):
        ep["steps"] = [[s, remap(f)] for s, f in ep["steps"]]
    ep["markers"] = [[s, remap(f)] for s, f in ep.get("markers", [])]
    ep["look_dir"] = str((src_dir / ep["look_dir"]).resolve())
    for v in (ep.get("retexture") or {}).values():
        if v.get("image"):
            v["image"] = str((src_dir / v["image"]).resolve())
    ep["hide_from_frame"] = {k: remap(f) for k, f in ep.get("hide_from_frame", {}).items()}
    ep["recovery_episode"] = True
    (out / "episode.json").write_text(json.dumps(ep, indent=1))
    return {"robot_skin_max_err_vs_demo": float(max(check)) if check else None}
