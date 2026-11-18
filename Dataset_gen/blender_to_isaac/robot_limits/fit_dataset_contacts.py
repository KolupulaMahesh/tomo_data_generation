"""Second step after fit_dataset_to_limits.py: keep the THUMBS clear of what they may not touch - numbers only.

The robot's console stops a run when two of its parts touch (its collision shapes; 2026-10-25: the left thumb and index
finger, 6.4 mm apart in the demos, closed up on the robot). In the full-sequence demos the thumbs come too close:
left thumb vs left index (PLACE_PHONE), left thumb vs the right hand (FOLD_TAPE: the right hand presses the tape down
onto the left thumb), right thumb vs its own palm (FOLD_TAPE). Here, per episode:
  * every frame where a thumb link (x_link_11/12/13) is nearer than TARGET_MM to a link it may not touch (SRDF), and
    nearer than it is with every joint at 0, is flagged; flagged frames of one hand within 2 x RAMP of each other form
    one stretch
  * for each stretch ONE fix: the smallest change of that thumb's 3 joints (same direction for the whole stretch), plus,
    when the thumb is too near the OTHER hand, moving its own hand straight away from it (the arm re-solved, the hand's
    turn kept) - smallest by thumb degrees + hand millimetres
  * the fix is eased in and out over RAMP frames (no jumps), then every frame is checked again
Every joint stays inside the 2-deg margin. Other files are hard-linked. Writes <dst>/contact_report.json.

    ../env/bin/python fit_dataset_contacts.py <src (the joint-fixed copy)> <dst>
"""
import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.argv_saved, sys.argv = sys.argv, [sys.argv[0]]
import check_motion as cm                                        # noqa: E402  (the robot's collision shapes + FK)
sys.argv = sys.argv_saved

fcl = cm.fcl
LIM = cm.LIMITS["joints"]
TARGET_MM, RAMP, MARGIN_DEG = 10.0, 10, 2.0
MAX_THUMB_DEG, HAND_MM = 15, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# ONLY the parts each thumb collides with in the full-sequence demos (2026-10-26 check): the left thumb with the left
# index finger (PLACE_PHONE) and with the right hand (FOLD_TAPE), the right thumb with its own palm. Everything else is
# left exactly as it is.
WATCH = {"LF": ("LF_link_2", "Right_hand"), "RF": ("Right_hand",)}
DIRS = [np.array(d) for d in np.ndindex(3, 3, 3) if any(v != 1 for v in d)]
DIRS = [d - 1 for d in DIRS]                                     # the 26 sign patterns of the 3 thumb joints


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = math.acos(c)
    return np.zeros(3) if th < 1e-9 else th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                                                                               R[1, 0] - R[0, 1]])


class Contacts:
    def __init__(self, names):
        self.names, self.R = names, cm.Robot()
        self.idx = {}
        for i, o in enumerate(self.R.objs):
            self.idx.setdefault(o["link"], []).append(i)
        self.thumb = {h: [f"{h}_link_1{k}" for k in (1, 2, 3)] for h in ("LF", "RF")}
        self.side = {"LF": "Left", "RF": "Right"}
        # the pairs a thumb link may not touch (SRDF), from the checker's pair list
        self.pairs = {h: [] for h in self.thumb}
        for a, b in zip(self.R.pa, self.R.pb):
            la, lb = self.R.objs[a]["link"], self.R.objs[b]["link"]
            for h, th in self.thumb.items():
                if la in th and lb not in th and lb.startswith(WATCH[h]):
                    self.pairs[h].append((a, b))
                elif lb in th and la not in th and la.startswith(WATCH[h]):
                    self.pairs[h].append((b, a))
        m = math.radians(MARGIN_DEG)
        self.lo = np.array([LIM[n]["lower_rad"] + m for n in names])
        self.hi = np.array([LIM[n]["upper_rad"] - m for n in names])
        self.tj = {h: [names.index(f"{h}_joint_1{k}") for k in (1, 2, 3)] for h in self.thumb}
        self.arm = {h: [names.index(f"{self.side[h]}_arm_joint{k}") for k in range(1, 8)] for h in self.thumb}
        self.memo = {}

    def _pose(self, q):
        """place every collision shape; -> their world bounding boxes (lo, hi) for a quick 'far apart' test."""
        W = self.R.link_poses(dict(zip(self.names, q)))
        lo, hi = np.empty((len(self.R.objs), 3)), np.empty((len(self.R.objs), 3))
        for i, o in enumerate(self.R.objs):
            T = W[o["link"]] @ o["T"]
            o["obj"].setTransform(fcl.Transform(T[:3, :3], T[:3, 3]))
            c = o["corners"] @ T[:3, :3].T + T[:3, 3]
            lo[i], hi[i] = c.min(0), c.max(0)
        return lo, hi

    def target(self, a, b):
        pair = tuple(sorted((self.R.objs[a]["link"], self.R.objs[b]["link"])))
        rest = self.R.rest_gap.get(pair, np.inf)
        return min(TARGET_MM / 1000, rest - 0.001)

    def too_near(self, q, h):
        """-> list of (thumb obj, other obj, gap m) nearer than their target, for hand h."""
        key = (h, q.tobytes())
        if key in self.memo:
            return self.memo[key]
        lo, hi = self._pose(q)
        out = []
        for a, b in self.pairs[h]:
            t = self.target(a, b)
            if t <= 0 or np.any(lo[a] > hi[b] + t) or np.any(lo[b] > hi[a] + t):     # boxes farther apart than t
                continue
            oa, ob = self.R.objs[a]["obj"], self.R.objs[b]["obj"]
            if fcl.collide(oa, ob, fcl.CollisionRequest(), fcl.CollisionResult()):
                out.append((a, b, 0.0))
                continue
            d = fcl.distance(oa, ob, fcl.DistanceRequest(), fcl.DistanceResult())
            if d < t:
                out.append((a, b, d))
        self.memo[key] = out
        return out

    def other_hand(self, h, b):
        link = self.R.objs[b]["link"]
        o = "RF" if h == "LF" else "LF"
        return link.startswith((o + "_", self.side[o] + "_"))

    def move_hand(self, q, h, away, d_mm):
        """the arm of hand h re-solved so its hand is d_mm further along `away`, turned as before."""
        if d_mm == 0:
            return q
        a = self.arm[h]
        hand = f"{self.side[h]}_hand"
        names = [self.names[i] for i in a]
        T0 = self.R.link_poses(dict(zip(self.names, q)))[hand]
        tgt = T0.copy()
        tgt[:3, 3] += away * d_mm / 1000
        fk = lambda x: self.R.link_poses({**dict(zip(self.names, q)), **dict(zip(names, x))})[hand]

        def err(x):
            T = fk(x)
            return np.concatenate([(T[:3, 3] - tgt[:3, 3]) * 100, rotvec(T[:3, :3] @ tgt[:3, :3].T) * 10,
                                   1e-3 * np.degrees(x - q[a])])
        s = least_squares(err, np.clip(q[a], self.lo[a], self.hi[a]), bounds=(self.lo[a], self.hi[a]))
        qq = q.copy()
        qq[a] = s.x
        return qq

    def apply(self, q, h, dirv, s_deg, d_mm, away, w):
        key = (h, q.tobytes(), tuple(dirv), s_deg, d_mm, away.tobytes(), w)    # frames repeat across episodes
        if key not in self.memo:
            qq = self.move_hand(q, h, away, d_mm * w) if d_mm else q.copy()
            j = self.tj[h]
            qq[j] = np.clip(qq[j] + np.radians(dirv * s_deg * w), self.lo[j], self.hi[j])
            self.memo[key] = qq
        return self.memo[key].copy()

    def solve(self, Q, h, frames):
        """ONE fix (dir, s_deg, d_mm) that clears every frame of the stretch; smallest s_deg + d_mm."""
        flagged = [f for f in frames if self.too_near(Q[f], h)]
        cross = any(self.other_hand(h, b) for f in flagged for _, b, _ in self.too_near(Q[f], h))
        aways = {}
        for f in frames:
            W = self.R.link_poses(dict(zip(self.names, Q[f])))
            o = "Right" if h == "LF" else "Left"
            v = W[f"{self.side[h]}_hand"][:3, 3] - W[f"{o}_hand"][:3, 3]
            aways[f] = v / np.linalg.norm(v)
        probe = flagged[::3] + [flagged[-1]]
        best = None
        for d in (HAND_MM if cross else [0]):
            if best and d >= best[0]:
                break
            moved = {f: self.move_hand(Q[f], h, aways[f], d) for f in probe}
            for s in range(1, MAX_THUMB_DEG + 1):
                if best and s + d >= best[0]:
                    break
                for dirv in DIRS:
                    ok = True
                    for f in probe:
                        qq = moved[f].copy()
                        j = self.tj[h]
                        qq[j] = np.clip(qq[j] + np.radians(dirv * s), self.lo[j], self.hi[j])
                        if self.too_near(qq, h):
                            ok = False
                            break
                    if ok:
                        best = (s + d, dirv, s, d)
                        break
                if best and best[3] == d:
                    break
        return best, aways


def stretches(flags, ramp):
    fr = np.flatnonzero(flags)
    out, start = [], None
    for i, f in enumerate(fr):
        start = f if start is None else start
        if i + 1 == len(fr) or fr[i + 1] - f > 2 * ramp:
            out.append((int(start), int(f)))
            start = None
    return out


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    a = ap.parse_args()
    if a.dst.exists():
        sys.exit(f"{a.dst} exists: pick a new folder")
    eps = sorted(e for e in a.src.iterdir() if e.is_dir() and (e / "q.npy").exists())
    names = json.loads((eps[0] / "joint_names.json").read_text())
    C = Contacts(names)
    a.dst.mkdir(parents=True)
    for p in a.src.iterdir():
        if p.is_file():
            link_or_copy(p, a.dst / p.name)
    solved, report = {}, {"source": str(a.src), "target_mm": TARGET_MM, "ramp_frames": RAMP, "watch": WATCH, "episodes": {}}
    first = None                                                 # (Q0, change, changed rows, changed joints, rep) of eps[0]
    for k, e in enumerate(eps):
        Q0 = np.load(e / "q.npy").astype(np.float64)
        Q = Q0.copy()
        rep = []
        if first is not None and Q0.shape == first[0].shape \
                and np.array_equal(Q0[np.ix_(first[2], first[3])], first[0][np.ix_(first[2], first[3])]):
            Q = Q0 + first[1]                                    # same joints there: the same fix (computed once)
            rep = first[4]
        for h in ("LF", "RF") if not rep else ():
            flags = np.array([bool(C.too_near(Q0[f], h)) for f in range(len(Q0))])
            for s0, s1 in stretches(flags, RAMP):
                frames = list(range(s0, s1 + 1))
                key = (h, Q0[s0:s1 + 1].tobytes())
                if key not in solved:
                    solved[key] = C.solve(Q0, h, frames)
                best, aways = solved[key]
                pairs = sorted({tuple(sorted((C.R.objs[x]["link"], C.R.objs[y]["link"])))
                                for f in frames for x, y, _ in C.too_near(Q0[f], h)})
                if best is None:
                    rep.append({"hand": h, "frames": [s0, s1], "pairs": pairs, "fixed": False})
                    continue
                _, dirv, s_deg, d_mm = best
                lo_f, hi_f = max(0, s0 - RAMP), min(len(Q) - 1, s1 + RAMP)
                for f in range(lo_f, hi_f + 1):
                    w = 1.0 if s0 <= f <= s1 else 0.5 - 0.5 * math.cos(
                        math.pi * ((f - lo_f) / RAMP if f < s0 else (hi_f - f) / RAMP))
                    away = aways.get(f, aways[s0] if f < s0 else aways[s1])
                    Q[f] = C.apply(Q[f], h, dirv, s_deg, d_mm, away, w)
                left = [f for f in range(lo_f, hi_f + 1) if C.too_near(Q[f], h)]
                rep.append({"hand": h, "frames": [s0, s1], "pairs": pairs, "fixed": not left,
                            "thumb_deg": [int(v) * s_deg for v in dirv], "hand_moved_mm": d_mm,
                            "still_near_frames": left[:10]})
        if first is None:                                        # the fix to reuse for episodes with the same joints there
            D = Q - Q0
            rows, cols = np.flatnonzero(np.abs(D).max(1) > 0), np.flatnonzero(np.abs(D).max(0) > 0)
            first = (Q0, D, rows, cols, rep)
        # every joint still inside the margin; jumps
        assert np.all(Q >= C.lo - 1e-6) and np.all(Q <= C.hi + 1e-6), e.name   # (float32 storage rounding)
        extra = float(np.degrees(np.abs(np.diff(Q, axis=0)) - np.abs(np.diff(Q0, axis=0))).max())
        report["episodes"][e.name] = {"stretches": rep, "extra_step_max_deg": round(extra, 2),
                                      "changed_frames": int((np.abs(Q - Q0).max(1) > 1e-9).sum())}
        (a.dst / e.name).mkdir()
        for p in e.iterdir():
            if p.name not in ("q.npy", "labels_raw.npz"):
                link_or_copy(p, a.dst / e.name / p.name)
        np.save(a.dst / e.name / "q.npy", Q.astype(np.load(e / "q.npy", mmap_mode="r").dtype))
        if (e / "labels_raw.npz").exists():
            raw = dict(np.load(e / "labels_raw.npz", allow_pickle=True))
            if "q" in raw and raw["q"].shape == Q.shape:
                raw["q"] = Q.astype(raw["q"].dtype)
            np.savez(a.dst / e.name / "labels_raw.npz", **raw)
        if k % 25 == 0 or k == len(eps) - 1:
            print(f"[contacts] {k + 1}/{len(eps)} {e.name}: " + "; ".join(
                f"{r['hand']} {r['frames'][0]}-{r['frames'][1]} "
                + (f"thumb {r['thumb_deg']} deg + hand {r['hand_moved_mm']} mm" if "thumb_deg" in r else "NOT FIXED")
                + ("" if r["fixed"] else " (still near!)") for r in rep) + f" | extra step {extra:.1f} deg", flush=True)
    (a.dst / "contact_report.json").write_text(json.dumps(report, indent=1, default=str))
    bad = [n for n, r in report["episodes"].items() if any(not s["fixed"] for s in r["stretches"])]
    print(f"[contacts] done: {len(eps)} episodes -> {a.dst}; not fully fixed in {len(bad)} {bad[:8]}; "
          f"largest extra step {max(r['extra_step_max_deg'] for r in report['episodes'].values()):.1f} deg")


if __name__ == "__main__":
    main()
