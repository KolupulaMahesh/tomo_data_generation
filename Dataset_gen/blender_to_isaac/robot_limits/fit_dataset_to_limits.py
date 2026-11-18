"""Make a copy of an Isaac-generator dataset whose JOINTS keep the real robot's limits - numbers only, no re-render.

Every joint at every frame of every episode is brought MARGIN_DEG inside its limit (tomo_re_uhand_limits.yaml; --scene <name>: that scene's
robot, scene_robot.py, 2026-11-09):
  * arm (7 joints): a joint past the margin is moved onto it - eased in and out over RAMP frames, never less than
    needed - and the arm's OTHER joints are re-solved so the hand stays exactly where it was (position + orientation),
    so the pictures and what the hand holds (the phone) stay right; of the arm shapes that do, the one nearest the
    original. (2026-10-26: holding only the wrist roll moved the phone-holding cup 9.5 mm; the physics replay then
    placed the phone 10 deg off.)
  * fingers and head: set onto the margin (a finger at its curl limit: 2 deg less curl, ~1-2 mm at the fingertip).
  * an episode where keeping the hand in place needs a visibly different arm (MAX_ELBOW_CM / MAX_EXTRA_STEP_DEG) is
    LEFT OUT of the copy, with the reason in the report.
Every other file is hard-linked (no copy, no disk use); only each episode's q.npy (and the joints in labels_raw.npz)
are new. The source is not changed.
Writes <dst>/fit_report.json: what changed, per joint and stage, and the largest hand error.

    ../env/bin/python fit_dataset_to_limits.py <src dataset> <dst dataset> [--margin-deg 2] [--scene <name>]
"""
import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                     # kinematics/ = a frozen copy of the brain's FK
from kinematics.fk import RobotFK                                # noqa: E402  (the brain's FK)
sys.path.insert(0, str(HERE.parent))
from scene_robot import scene_setup, urdf_limits                 # noqa: E402

LIMITS = yaml.safe_load((HERE / "tomo_re_uhand_limits.yaml").read_text())["joints"]
URDF = HERE / "source/robot_description.urdf"                    # the robot's own (same kinematics as the brain's)
HAND_TOL_CM, HAND_TOL_DEG = 0.1, 0.1                             # a re-solve must keep the hand this close
# A 7-joint arm can hold the hand in place in many shapes: of those, take the one NEAREST the original (this weight per
# degree a re-solved joint moves). Without it the solver could pick a far one: elbow up to 9.8 cm away and a 41-deg jump
# between two frames (first build, 2026-10-26).
REG_PER_DEG = 1e-3
# An episode is LEFT OUT when keeping its hand in place needs a visibly different arm: the elbow moved more than this, or a
# frame-to-frame joint step this much bigger than the original (13 of 250 full-sequence episodes, right wrist at its
# limit in an awkward pose at frames 179-198: elbow up to 9.7 cm). READY's 3.0 cm (arms out of view) stays in.
MAX_ELBOW_CM, MAX_EXTRA_STEP_DEG = 3.5, 5.0
# A limit change is eased in and out over this many frames (a change that switched on in one frame made the re-solved
# joints jump: +6.9 deg steps at frame 277, 2026-10-26).
RAMP = 10
# The wrist roll (joint7) sits ON its limit in poses whose hand turn needs exactly that wrist value: keeping the whole
# hand pose then moves the elbow up to 14.7 cm. So when the wrist is moved in, the CUP point (between the 4 fingertips,
# where the hand holds the phone / touches things) is kept exactly and the hand's turn only loosely (it turns ~ the
# wrist change, <= ~2 deg). Holding the wrist alone (nothing re-solved) moved that point 9.5 mm: phone placed 10 deg off.
WRIST, TURN_WEIGHT = 6, 0.3


def ease(off, r):
    """per joint: the needed change spread over r frames each side with a cosine taper, never less than needed."""
    w = 0.5 + 0.5 * np.cos(np.pi * np.arange(-r, r + 1) / (r + 1))           # 1 at 0, -> 0 at +-(r+1)
    out = np.zeros_like(off)
    for sign in (1.0, -1.0):
        mag = np.maximum(sign * off, 0.0)
        pad = np.pad(mag, ((r, r), (0, 0)))
        best = np.max(np.stack([pad[k:k + len(off)] * w[k] for k in range(2 * r + 1)]), axis=0)
        out += sign * best
    return out


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(c)
    return np.zeros(3) if th < 1e-9 else th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                                                                               R[1, 0] - R[0, 1]])


class Fitter:
    def __init__(self, names, margin, urdf=URDF, limits=LIMITS, cup_links="{hand}_link_{k}endeff"):
        self.names, self.fk, self.cup_links = names, RobotFK(str(urdf)), cup_links
        m = math.radians(margin)
        self.lo = np.array([limits[n]["lower_rad"] + m for n in names])
        self.hi = np.array([limits[n]["upper_rad"] - m for n in names])
        self.arms = {s: [names.index(f"{s}_arm_joint{k}") for k in range(1, 8)] for s in ("Left", "Right")}
        armset = {i for a in self.arms.values() for i in a}
        self.rest = [i for i in range(len(names)) if i not in armset]       # head + fingers: held on the margin
        self.memo = {}

    def hand(self, side, q_arm):
        return self.fk.link_transform(f"{side}_hand", dict(zip([self.names[i] for i in self.arms[side]], q_arm)))

    def link(self, side, link, q_arm):
        return self.fk.link_transform(link, dict(zip([self.names[i] for i in self.arms[side]], q_arm)))[:3, 3]

    def fit_arm(self, side, q_arm, target, cup_local):
        """target: q_arm with the eased limit changes; cup_local: the cup point in the hand frame. -> (new 7 joint
        values, cup error cm, hand turn deg, elbow moved cm); memoised (frames repeat across episodes)."""
        key = (side, q_arm.tobytes(), target.tobytes(), cup_local.tobytes())
        if key in self.memo:
            return self.memo[key]
        a = self.arms[side]
        lo, hi = self.lo[a], self.hi[a]
        out_ = np.abs(target - q_arm) > 1e-12                   # moved towards the inside: held there
        qc = np.where(out_, target, q_arm)
        if not out_.any():
            res = (q_arm, 0.0, 0.0, 0.0)
        else:
            T0 = self.hand(side, q_arm)
            cup0 = T0[:3, :3] @ cup_local + T0[:3, 3]            # the cup point stays exactly where it was
            turn_w = TURN_WEIGHT if out_[WRIST] else 10.0       # the hand's turn: loose when the wrist moved in
            free = np.flatnonzero(~out_)

            def err(x):
                qq = qc.copy()
                qq[free] = x
                T = self.hand(side, qq)
                return np.concatenate([(T[:3, :3] @ cup_local + T[:3, 3] - cup0) * 100,
                                       (T[:3, 3] - T0[:3, 3]) * (0 if out_[WRIST] else 100),
                                       rotvec(T[:3, :3] @ T0[:3, :3].T) * turn_w,
                                       REG_PER_DEG * np.degrees(x - q_arm[free])])
            s = least_squares(err, np.clip(qc[free], lo[free], hi[free]), bounds=(lo[free], hi[free]),
                              xtol=1e-12, ftol=1e-12, gtol=1e-12)
            qq = qc.copy()
            qq[free] = s.x
            T = self.hand(side, qq)
            elbow = f"{side}_arm_link4"
            res = (qq, float(100 * np.linalg.norm(T[:3, :3] @ cup_local + T[:3, 3] - cup0)),
                   float(math.degrees(np.linalg.norm(rotvec(T[:3, :3] @ T0[:3, :3].T)))),
                   float(100 * np.linalg.norm(self.link(side, elbow, qq) - self.link(side, elbow, q_arm))))
        self.memo[key] = res
        return res

    def fit(self, q):
        """-> (fitted q, [hand cm, hand deg] worst, elbow moved cm per frame)."""
        q = q.astype(np.float64).copy()
        worst, elbow = [0.0, 0.0], np.zeros(len(q))
        for side, a in self.arms.items():
            qa = q[:, a].copy()
            target = qa + ease(np.clip(qa, self.lo[a], self.hi[a]) - qa, RAMP)
            h = "LF" if side == "Left" else "RF"
            tips = [self.cup_links.format(hand=h, k=k) for k in (2, 3, 4, 5)]
            for f in range(len(q)):
                cup_local = np.zeros(3)
                if np.any(np.abs(target[f] - qa[f]) > 1e-12):   # the cup point in the hand frame (fingers as recorded)
                    qd = dict(zip(self.names, q[f]))
                    T = self.fk.link_transform(f"{side}_hand", qd)
                    cup = np.mean([self.fk.link_transform(t, qd)[:3, 3] for t in tips], 0)
                    cup_local = np.round(T[:3, :3].T @ (cup - T[:3, 3]), 9)
                q[f, a], ecm, edeg, el = self.fit_arm(side, qa[f], target[f], cup_local)
                worst = [max(worst[0], ecm), max(worst[1], edeg)]
                elbow[f] = max(elbow[f], el)
        q[:, self.rest] = np.clip(q[:, self.rest], self.lo[self.rest], self.hi[self.rest])
        return q, worst, elbow


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--margin-deg", type=float, default=2.0)
    ap.add_argument("--scene", help="this scene's robot (default: the first robot)")
    ap.add_argument("--max-extra-step-deg", type=float, default=MAX_EXTRA_STEP_DEG,
                    help="leave an episode out when a frame-to-frame joint step grows by more than this")
    a = ap.parse_args()
    if a.dst.exists():
        sys.exit(f"{a.dst} exists: pick a new folder (the source is never changed)")
    eps = sorted(e for e in a.src.iterdir() if e.is_dir() and (e / "q.npy").exists())
    names = json.loads((eps[0] / "joint_names.json").read_text())
    setup = scene_setup(a.scene) if a.scene else None
    fitter = Fitter(names, a.margin_deg, setup["urdf"], urdf_limits(setup["urdf"], 0.0), setup["cup_links"]) if setup else Fitter(names, a.margin_deg)
    a.dst.mkdir(parents=True)
    for p in a.src.iterdir():                                    # dataset-level files (task.yaml, ...)
        if p.is_file():
            link_or_copy(p, a.dst / p.name)
    changes = defaultdict(lambda: {"episodes": 0, "frames": 0, "max_change_deg": 0.0})
    per_stage = defaultdict(lambda: {"elbow_moved_max_cm": 0.0, "extra_step_max_deg": 0.0})
    worst, left_out = [0.0, 0.0], {}
    for k, e in enumerate(eps):
        if json.loads((e / "joint_names.json").read_text()) != names:
            sys.exit(f"{e.name}: another joint order than {eps[0].name}")
        q0 = np.load(e / "q.npy")
        q1, w, elbow = fitter.fit(q0)
        worst = [max(worst[0], w[0]), max(worst[1], w[1])]
        segs = json.loads((e / "segments.json").read_text())["segments"]
        stage = np.empty(len(q0), object)
        for s in segs:
            stage[s["start"]:s["end"] + 1] = s["name"]
        # jumps: how much bigger the largest frame-to-frame joint step got than in the original
        extra = np.degrees(np.abs(np.diff(q1, axis=0)) - np.abs(np.diff(q0.astype(np.float64), axis=0))).max(1)
        bad = np.flatnonzero((elbow[1:] > MAX_ELBOW_CM) | (extra > a.max_extra_step_deg)) + 1
        if len(bad):
            left_out[e.name] = {"frames": [int(bad[0]), int(bad[-1])], "stages": sorted(set(stage[bad])),
                                "elbow_moved_cm": round(float(elbow.max()), 2), "extra_step_deg": round(float(extra.max()), 2)}
            print(f"[fit] {e.name} LEFT OUT: keeping the hand needs a visibly different arm at frames {bad[0]}-{bad[-1]} "
                  f"(elbow {elbow.max():.1f} cm, step +{extra.max():.1f} deg)", flush=True)
            continue
        for st in set(stage):
            m = stage == st
            ps = per_stage[st]
            ps["elbow_moved_max_cm"] = round(max(ps["elbow_moved_max_cm"], float(elbow[m].max())), 2)
            ps["extra_step_max_deg"] = round(max(ps["extra_step_max_deg"], float(extra[m[1:]].max(initial=0.0))), 2)
        d = np.degrees(np.abs(q1 - q0.astype(np.float64)))
        for j in np.flatnonzero((d > 1e-4).any(0)):
            for st in sorted(set(stage[d[:, j] > 1e-4]), key=lambda s: [x["name"] for x in segs].index(s)):
                c = changes[(names[j], st)]
                c["episodes"] += 1
                c["frames"] += int(((d[:, j] > 1e-4) & (stage == st)).sum())
                c["max_change_deg"] = round(max(c["max_change_deg"], float(d[stage == st, j].max())), 2)
        (a.dst / e.name).mkdir()
        for p in e.iterdir():
            if p.name not in ("q.npy", "labels_raw.npz"):
                (shutil.copytree if p.is_dir() else link_or_copy)(p, a.dst / e.name / p.name)
        np.save(a.dst / e.name / "q.npy", q1.astype(q0.dtype))
        if (e / "labels_raw.npz").exists():                     # the generator's raw labels carry the joints too
            raw = dict(np.load(e / "labels_raw.npz", allow_pickle=True))
            if "q" in raw:
                raw["q"] = fitter.fit(raw["q"])[0].astype(raw["q"].dtype)
            np.savez(a.dst / e.name / "labels_raw.npz", **raw)
        if k % 25 == 0 or k == len(eps) - 1:
            print(f"[fit] {k + 1}/{len(eps)} {e.name}  hand kept within {worst[0]:.3f} cm / {worst[1]:.3f} deg", flush=True)
    rep = {"source": str(a.src), "margin_deg": a.margin_deg, "max_extra_step_deg": a.max_extra_step_deg, "episodes": len(eps) - len(left_out),
           "left_out": left_out, "hand_error_max": {"cm": round(worst[0], 4), "deg": round(worst[1], 4)},
           "per_stage": dict(per_stage),
           "changed": [{"joint": j, "stage": s, **c} for (j, s), c in changes.items()]}
    (a.dst / "fit_report.json").write_text(json.dumps(rep, indent=1))
    print(f"[fit] done: {len(eps) - len(left_out)} episodes -> {a.dst} ({len(left_out)} left out: {sorted(left_out)}); "
          f"hand kept within {worst[0]:.3f} cm / {worst[1]:.3f} deg "
          f"(limit {HAND_TOL_CM} cm / {HAND_TOL_DEG} deg) {'OK' if worst[0] <= HAND_TOL_CM and worst[1] <= HAND_TOL_DEG else 'CHECK!'}")
    print("[fit] per stage: elbow moved (max) | largest extra frame-to-frame step vs the original")
    for st, ps in per_stage.items():
        print(f"   {st:15s} {ps['elbow_moved_max_cm']:5.2f} cm | {ps['extra_step_max_deg']:5.2f} deg")
    for (j, s), c in sorted(changes.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"   {j:18s} {s:15s} {c['episodes']:>4d} episodes, {c['frames']:>6d} frames, up to {c['max_change_deg']:.2f} deg")


if __name__ == "__main__":
    main()
