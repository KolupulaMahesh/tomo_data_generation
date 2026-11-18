#!/usr/bin/env python3
"""Check a generated dataset BEFORE any training (2026-11-02, Praveen: "even before training we need to know"): a bad
dataset found after 10 hours of training is 10 hours lost. Uses only the numbers in the episodes (joints, object poses,
vacuums, step table, the generator's limits.json): no pictures, no model, no GPU; seconds, even for hundreds of episodes.
Read-only for the episodes. A FAIL means: don't train on this yet. PASS does not prove the model will work on the robot:
it only rules out what can be seen from the data.

    env/bin/python check_dataset.py <dataset folder> [--reference <other dataset>] [--out DIR]

Writes data_check.txt + data_check.json into the dataset folder (or DIR); the training's episode check ignores them.
Exit code 0 = no check failed, 1 = at least one failed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "robot_limits"))             # its pure-numpy kinematics
from kinematics import RobotFK                             # noqa: E402

URDF = HERE / "robot_limits/source/robot_description.urdf"
BODY = ("Right_arm", "Left_arm", "Head")
TIPS = {"suction": [f"RF_link_{i}endeff" for i in (2, 3, 4, 5)],        # what touches the object, per vacuum
        "left_suction": [f"LF_link_{i}endeff" for i in (2, 3, 4, 5)]}
OBJECTS = {"phone": "phone_world.npy", "box": "box_world.npy"}

# ── the rules: what counts as a problem ────────────────────────────────────────────────────────────────────────────
IDENTICAL_DEG = 0.01        # head + arm joints closer than this between all episodes = the same motion at that frame
MAX_IDENTICAL_PCT = 50.0    # more of the frames than this are the same motion in every episode -> FAIL
MIN_EXPLAINED_PCT = 50.0    # a step's differences between episodes must be explained this much by the objects' start
MAX_GRAB_SPREAD_MM = 2.0    # where the hand touches an object (in the object's own frame) may differ this much
NEAR_OBJECT_MM = 150.0      # a vacuum switching on farther than this from every tracked object: object not tracked
COPY_FRAMES = 30            # "just copy the current joints": the error this many frames ahead...
MIN_COPY_DEG = 1.0          # ...below this in a step, a model can score well there without learning the motion
DEAD_JOINT_DEG = 0.5        # a joint moving less than this in the whole dataset is never taught
MAX_STEP_RATIO = 10.0       # the longest step more than this many times the shortest: the short one is rarely sampled
MOVE_MM = 0.5               # an object moving more than this per frame counts as moving
MAX_FOLDS = 30              # leave-one-episode-out: at most this many episodes left out, spread evenly


def load(dataset):
    """the episodes' numbers; files missing or broken -> named in `problems`, the episode is left out."""
    eps, problems = [], []
    for d in sorted(p for p in Path(dataset).glob("fs_*") if p.is_dir()):
        try:
            names = json.loads((d / "joint_names.json").read_text())
            q = np.load(d / "q.npy").astype(np.float64)
            seg = json.loads((d / "segments.json").read_text())["segments"]
            lab = np.load(d / "labels.npz", allow_pickle=True)
            vac = {"suction": lab["suction"] if "suction" in lab.files else np.load(d / "suction.npy").ravel()}
            if "left_suction" in lab.files:
                vac["left_suction"] = lab["left_suction"]
            obj = {k: np.load(d / f).astype(np.float64) for k, f in OBJECTS.items() if (d / f).exists()}
        except (OSError, KeyError, ValueError) as e:
            problems.append(f"{d.name}: {e}")
            continue
        bad = [k for k, v in {"joints": q, **obj}.items() if not np.isfinite(v).all()]
        if bad:
            problems.append(f"{d.name}: NaN / inf in {', '.join(bad)}")
            continue
        if any(len(v) != len(q) for v in [*vac.values(), *obj.values()]):
            problems.append(f"{d.name}: joints, vacuums and objects differ in length")
            continue
        lim = d / "limits.json"
        eps.append(dict(name=d.name, names=names, q=q, seg=seg, vac={k: np.asarray(v).ravel() > 0.5 for k, v in vac.items()},
                        obj=obj, limits=json.loads(lim.read_text()) if lim.exists() else None))
    return eps, problems


def align(eps):
    """every episode on one timeline, step by step (each step at its median length; the same frames when the episodes
    already share a timeline). -> (step names, [(first, last) per step], index table per episode)."""
    names = [s["name"] for s in eps[0]["seg"]]
    lens = [int(np.median([s["end"] - s["start"] + 1 for e in eps for s in e["seg"] if s["name"] == n])) for n in names]
    spans, a = [], 0
    for n in lens:
        spans.append((a, a + n - 1))
        a += n
    idx = [np.concatenate([np.round(np.linspace(s["start"], s["end"], n)).astype(int) for s, n in zip(e["seg"], lens)])
           for e in eps]
    return names, spans, idx


def pose_xyyaw(M):
    return [M[0, 3], M[1, 3], np.arctan2(M[1, 0], M[0, 0])]


def ranges(frames):
    """[3, 4, 5, 9] -> '3-5, 9'"""
    if not len(frames):
        return ""
    cut = np.flatnonzero(np.diff(frames) > 1)
    a, b = np.r_[frames[0], np.asarray(frames)[cut + 1]], np.r_[np.asarray(frames)[cut], frames[-1]]
    return ", ".join(f"{x}" if x == y else f"{x}-{y}" for x, y in zip(a, b))


# ── the checks: each returns (verdict PASS / WARN / FAIL / N/A, lines, numbers) ────────────────────────────────────
def check_files(eps, problems, total):
    same = len({tuple(e["names"]) for e in eps}) == 1 and len({tuple(s["name"] for s in e["seg"]) for e in eps}) == 1
    lines = [f"{len(eps)} of {total} episodes usable; {min(len(e['q']) for e in eps)}-{max(len(e['q']) for e in eps)} "
             f"frames, {len(eps[0]['names'])} joints, {len(eps[0]['seg'])} steps"] + problems
    if not same:
        lines.append("the episodes differ in joints or steps")
    return ("FAIL" if problems or not same else "PASS"), lines, {"usable": len(eps), "total": total}


def check_robot(eps):
    have = [e for e in eps if e["limits"]]
    if not have:
        return "N/A", ["no limits.json in the episodes: not checked (robot_limits/check_motion.py <dataset> can do it)"], {}
    out, coll, fast = {}, {}, {}
    for e in have:
        for j, v in e["limits"].get("position", {}).items():
            if v.get("outside"):
                o = out.setdefault(j, dict(eps=0, frames=v.get("outside_frames"), lo=v["min_deg"], hi=v["max_deg"],
                                           limits=v["limits_deg"]))
                o["eps"] += 1
                o["lo"], o["hi"] = min(o["lo"], v["min_deg"]), max(o["hi"], v["max_deg"])
        for p, v in e["limits"].get("collision", {}).items():
            c = coll.setdefault(p, dict(eps=0, where=v.get("where")))
            c["eps"] += 1
        for j, v in e["limits"].get("speed", {}).items():
            if v.get("over_cap"):
                fast[j] = max(fast.get(j, 0), v["max_deg_s"])
    lines = [f"{len(have)} of {len(eps)} episodes have the report"]
    lines += [f"outside its limit: {j} in {o['eps']} episodes, frames {o['frames']}, went {o['lo']:.1f}..{o['hi']:.1f} deg, "
              f"limits {o['limits'][0]:.1f}..{o['limits'][1]:.1f}" for j, o in sorted(out.items())]
    lines += [f"touching: {p} in {c['eps']} episodes, frames {c['where']}" for p, c in sorted(coll.items())]
    if fast:
        j = max(fast, key=fast.get)
        lines.append(f"speed (not a fail, the robot node paces speed): {len(fast)} joints over the cap, fastest {j} "
                     f"{fast[j]:.0f} deg/s")
    return ("FAIL" if out or coll else "PASS"), lines, {"outside": out, "touching": coll, "over_speed_cap": fast}


def check_identical(Q, body, names, spans):
    spread = np.degrees(Q[:, :, body].std(0)).max(1)                     # per frame: the joint differing most between episodes
    same = spread < IDENTICAL_DEG
    pct = 100 * same.mean()
    lines = [f"{same.sum()} of {len(same)} frames ({pct:.0f}%): every episode moves exactly the same, so what the camera "
             f"sees cannot change what the robot should do there"]
    per = {}
    for n, (a, b) in zip(names, spans):
        per[n] = round(100 * float(same[a:b + 1].mean()), 1)
        lines.append(f"  {n:18s} {b - a + 1:5d} frames, {per[n]:5.1f}% identical, biggest difference between episodes "
                     f"{spread[a:b + 1].max():6.2f} deg")
    return ("FAIL" if pct > MAX_IDENTICAL_PCT else "PASS"), lines, {"identical_pct": round(float(pct), 1), "per_step": per}


def check_explained(Q, X, body, names, spans):
    """how much of the differences between episodes the objects' start positions explain: replay the average motion of
    the other episodes vs a straight-line fit on the objects' start, on an episode left out."""
    n = len(Q)
    if X.shape[1] == 0:
        return "N/A", ["no tracked object starts at a different place in different episodes"], {}
    if n < X.shape[1] + 3:
        return "N/A", [f"{n} episodes are too few for {X.shape[1]} object numbers"], {}
    A = np.c_[np.ones(n), (X - X.mean(0)) / X.std(0)]
    folds = range(n) if n <= MAX_FOLDS else np.linspace(0, n - 1, MAX_FOLDS).astype(int)
    lines, per, low = [], {}, []
    for name, (a, b) in zip(names, spans):
        Y = Q[:, a:b + 1][:, :, body]
        if np.degrees(Y.std(0)).max() < IDENTICAL_DEG:
            continue
        em, ef = [], []
        for h in folds:
            tr = np.arange(n) != h
            em.append(np.abs(Y[tr].mean(0) - Y[h]).mean())
            B = np.linalg.lstsq(A[tr], Y[tr].reshape(tr.sum(), -1), rcond=1e-6)[0]
            ef.append(np.abs((A[h] @ B).reshape(Y[h].shape) - Y[h]).mean())
        em, ef = np.degrees(np.mean(em)), np.degrees(np.mean(ef))
        pct = 100 * (1 - ef / em) if em > 0 else 0.0
        per[name] = dict(replay_deg=round(float(em), 3), fit_deg=round(float(ef), 3), explained_pct=round(float(pct), 1))
        lines.append(f"  {name:18s} replaying the average: {em:5.2f} deg | with the objects' start: {ef:5.2f} deg | "
                     f"{pct:3.0f}% explained")
        if pct < MIN_EXPLAINED_PCT:
            low.append(name)
    if not per:
        return "N/A", ["no step differs between episodes"], {}
    if low:
        lines.append(f"differences the objects don't explain in: {', '.join(low)} - a model cannot predict them from the camera")
    return ("FAIL" if low else "PASS"), lines, per


def check_grabs(eps, fk):
    """every vacuum switching on: where the hand is in the nearest tracked object's own frame, across the episodes. Wide
    spread = the hand does not follow the object (the object moved, the hands went to the old spot)."""
    use = [eps[i] for i in np.linspace(0, len(eps) - 1, min(len(eps), 40)).astype(int)]
    lines, data, bad = [], {}, False
    for vac in sorted({k for e in use for k in e["vac"]}):
        ons = [np.flatnonzero(np.diff(e["vac"][vac].astype(int)) > 0) + 1 for e in use]
        if len({len(o) for o in ons}) > 1:
            lines.append(f"{vac}: switches on {sorted({len(o) for o in ons})} times, differs between episodes")
            bad = True
            continue
        for k in range(len(ons[0])):
            fr = [int(o[k]) for o in ons]
            tips = np.array([np.mean([fk.link_transform(t, dict(zip(e["names"], map(float, e["q"][f]))))[:3, 3]
                                      for t in TIPS[vac]], 0) for e, f in zip(use, fr)])
            best = None
            for ob in use[0]["obj"]:
                loc = np.array([(np.linalg.inv(e["obj"][ob][f]) @ np.r_[p, 1])[:3] for e, f, p in zip(use, fr, tips)]) * 1000
                dist = float(np.median(np.linalg.norm(loc, axis=1)))
                if best is None or dist < best[1]:
                    best = (ob, dist, loc)
            label = f"{vac} on at frame {fr[0]}" + ("" if min(fr) == max(fr) else f"-{max(fr)}")
            if best is None or best[1] > NEAR_OBJECT_MM:
                lines.append(f"{label}: no tracked object within {NEAR_OBJECT_MM:.0f} mm, not checked")
                continue
            spread = float(np.linalg.norm(best[2] - best[2].mean(0), axis=1).max())
            data[label] = dict(object=best[0], spread_mm=round(spread, 3))
            lines.append(f"{label}: at the {best[0]}, varies {spread:.2f} mm between episodes")
            bad |= spread > MAX_GRAB_SPREAD_MM
    return ("FAIL" if bad else "PASS" if data else "N/A"), lines, data


def check_copying(Q, body, names, spans):
    F = Q.shape[1]
    lines, per = [], {}
    for n, (a, b) in zip(names, spans):
        b = min(b, F - 1 - COPY_FRAMES)
        if b >= a:
            per[n] = float(np.degrees(np.abs(Q[:, a:b + 1][:, :, body] - Q[:, a + COPY_FRAMES:b + 1 + COPY_FRAMES][:, :, body]).mean()))
    low = [n for n, v in per.items() if v < MIN_COPY_DEG]
    lines.append(f"copying the current joints is off by {min(per.values()):.1f}-{max(per.values()):.1f} deg, "
                 f"{COPY_FRAMES} frames ahead")
    if low:
        lines.append(f"almost no motion, a model can score well just by copying: {', '.join(low)}")
    return ("WARN" if low else "PASS"), lines, {k: round(v, 2) for k, v in per.items()}


def check_dead(Q, names):
    rng = np.degrees(Q.max((0, 1)) - Q.min((0, 1)))
    dead = [names[i] for i in range(len(names)) if rng[i] < DEAD_JOINT_DEG]
    return ("WARN" if dead else "PASS"), [f"{len(dead)} of {len(names)} joints never move, so they are never taught"
                                          + (f": {', '.join(dead)}" if dead else "")], {"never_move": dead}


def check_balance(eps, names, spans):
    lens = [b - a + 1 for a, b in spans]
    lines, data = [f"steps: shortest {min(lens)} frames ({names[int(np.argmin(lens))]}), longest {max(lens)} frames "
                   f"({names[int(np.argmax(lens))]})"], {"frames_per_step": dict(zip(names, lens))}
    warn = max(lens) > MAX_STEP_RATIO * min(lens)
    for vac in sorted({k for e in eps for k in e["vac"]}):
        v = [e["vac"][vac] for e in eps if vac in e["vac"]]
        sw = sorted({int(np.abs(np.diff(x.astype(int))).sum()) for x in v})
        pct = 100 * float(np.mean([x.mean() for x in v]))
        lines.append(f"{vac}: on in {pct:.0f}% of frames, switches {sw} times per episode")
        data[vac] = dict(on_pct=round(pct, 1), switches=sw)
        warn |= len(sw) > 1
    return ("WARN" if warn else "PASS"), lines, data


def check_moving(eps):
    """objects moving while no vacuum is on: held another way (fingers), falling, or a vacuum label that is wrong."""
    lines, data = [], {}
    for ob in eps[0]["obj"]:
        e = eps[0]
        p = e["obj"][ob][:, :3, 3]
        mv = np.linalg.norm(np.diff(p, axis=0), axis=1) * 1000 > MOVE_MM
        any_vac = np.any([v[:-1] for v in e["vac"].values()], axis=0)
        fr = np.flatnonzero(mv & ~any_vac)
        data[ob] = dict(moving_frames=int(mv.sum()), without_vacuum=int(len(fr)))
        lines.append(f"{ob}: moves in {int(mv.sum())} frames, {len(fr)} of them with no vacuum on"
                     + (f" ({ranges(fr)})" if len(fr) else ""))
    return "INFO", lines, data


def compare(eps, ref, names, spans, Q, body):
    """the same step table + vacuum labels as the reference dataset? what changed, before anyone trains on it."""
    rn, rs, ri = align(ref)
    lines, data = [], {}
    if rn != names:
        return "INFO", [f"other steps than the reference: {rn} vs {names}"], {}
    R = np.stack([e["q"][i] for e, i in zip(ref, ri)]) if [b - a for a, b in rs] == [b - a for a, b in spans] else None
    for n, (a, b) in zip(names, spans):
        if R is None:
            break
        d = float(np.degrees(np.abs(Q[:, a:b + 1][:, :, body].mean(0) - R[:, a:b + 1][:, :, body].mean(0)).mean()))
        data[n] = round(d, 3)
        if d > IDENTICAL_DEG:
            lines.append(f"  {n:18s} average motion differs {d:.2f} deg")
    for vac in sorted({k for e in eps for k in e["vac"]} | {k for e in ref for k in e["vac"]}):
        here = sorted({int(np.abs(np.diff(e["vac"][vac].astype(int))).sum()) for e in eps if vac in e["vac"]})
        there = sorted({int(np.abs(np.diff(e["vac"][vac].astype(int))).sum()) for e in ref if vac in e["vac"]})
        if here != there:
            lines.append(f"{vac}: switches {here or 'not labelled'} times per episode, the reference {there or 'not labelled'}")
    return "INFO", lines or ["no difference found"], data


def main():
    ap = argparse.ArgumentParser(description="Check a generated dataset before any training")
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--reference", type=Path, help="a dataset to compare with (labels, steps, average motion)")
    ap.add_argument("--out", type=Path, help="where data_check.txt / .json go (default: the dataset folder)")
    args = ap.parse_args()
    total = len([p for p in args.dataset.glob("fs_*") if p.is_dir()])
    eps, problems = load(args.dataset)
    if len(eps) < 2:
        sys.exit(f"[check] {args.dataset}: {len(eps)} usable episodes of {total}; at least 2 are needed. {problems[:5]}")
    names, spans, idx = align(eps)
    Q = np.stack([e["q"][i] for e, i in zip(eps, idx)])
    jn = eps[0]["names"]
    body = [i for i, n in enumerate(jn) if n.startswith(BODY)]
    X = np.array([[v for ob in sorted(e["obj"]) for v in pose_xyyaw(e["obj"][ob][0])] for e in eps])
    X = X[:, X.std(0) > 1e-6]                                            # only what really differs between episodes
    checks = [("Files complete and readable", check_files(eps, problems, total)),
              ("Robot can physically do it", check_robot(eps)),
              ("Camera can affect the action", check_identical(Q, body, jn, spans)),
              ("Objects explain the motion", check_explained(Q, X, body, names, spans)),
              ("Hand really reaches the object", check_grabs(eps, RobotFK(str(URDF)))),
              ("No copying shortcut", check_copying(Q, body, names, spans)),
              ("Every joint is taught", check_dead(Q, jn)),
              ("Steps and vacuums balanced", check_balance(eps, names, spans)),
              ("Objects moving without a vacuum", check_moving(eps))]
    if args.reference:
        ref, _ = load(args.reference)
        if ref:
            checks.append((f"Compared with {args.reference.name}", compare(eps, ref, names, spans, Q, body)))
    fails = [n for n, c in checks if c[0] == "FAIL"]
    warns = [n for n, c in checks if c[0] == "WARN"]
    out = [f"DATA CHECK  {args.dataset.resolve()}", ""]
    for n, (verdict, lines, _) in checks:
        out.append(f"[{verdict:4s}] {n}")
        out += [f"       {x}" for x in lines]
    out += ["", (f"RESULT: DON'T TRAIN YET - {len(fails)} check(s) failed: {', '.join(fails)}" if fails else
                 f"RESULT: no check failed ({len(warns)} warning(s)). This does not prove the model will work on the robot.")]
    text = "\n".join(out)
    print(text)
    dest = args.out or args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "data_check.txt").write_text(text + "\n")
    (dest / "data_check.json").write_text(json.dumps(
        {"dataset": str(args.dataset.resolve()), "failed": fails, "warnings": warns,
         "checks": {n: {"verdict": c[0], "lines": c[1], "numbers": c[2]} for n, c in checks}}, indent=1, default=float))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
