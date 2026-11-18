"""One episode: play the demo stage by stage; at the chosen stages cause a failure, notice it, choose a recovery, do it,
check it, and continue the demo from where the failure happened (not from the start).

    Observe -> Execute -> Verify -> next stage
    failure: Detect -> Classify -> Choose recovery -> Execute -> Verify -> (retry with another strategy | terminal)
             -> Resume the original task -> ... -> Final verification (validate.py)
"""
from __future__ import annotations

import numpy as np

from . import observe, scenarios as S
from .builder import Builder
from .detect import Grasp
from .clearance import Clearance
from .kin import angle_deg


def pick_category(fcfg, rng):
    mix = fcfg["episode_mix"]
    names = list(mix)
    p = np.array([mix[n] for n in names], float)
    return names[int(rng.choice(len(names), p=p / p.sum()))]


def plan(ctx, category):
    """the scenarios for this episode: available ones, at different stages, not overlapping."""
    rng, f = ctx.rng, ctx.f
    pool = []
    for name, cfg in f["scenarios"].items():
        if not cfg or not cfg.get("weight"):
            continue
        sc = S.make(ctx, name, cfg)
        if sc.available():
            pool.append((sc, float(cfg["weight"])))
    if category == "normal" or not pool:
        return [], pool
    k = 1 if category in ("single", "unrecoverable") else int(rng.integers(f["multi_count"][0], f["multi_count"][1] + 1))
    chosen = []
    order = rng.permutation(len(pool))
    w = np.array([pool[i][1] for i in order])
    order = order[np.argsort(-rng.random(len(order)) ** (1 / w))]       # weighted order
    for i in order:
        sc = pool[i][0]
        if any(not (sc.frame() >= o.end() or sc.end() <= o.frame()) for o in chosen):
            continue
        if any(o.key == sc.key and type(o) is type(sc) for o in chosen):
            continue
        chosen.append(sc)
        if len(chosen) == k:
            break
    return sorted(chosen, key=lambda s: s.frame()), pool


def handle(ctx, sc, unrecoverable):
    """one failure, from cause to resumed task. Returns (resume demo frame, terminal?)."""
    b, f, rng = ctx.b, ctx.f, ctx.rng
    names = list(b.s.objects)
    b.label(phase="failure", failure=sc.name, recovery="", attempt=0)
    start = b.n
    info = sc.inject(unrecoverable)
    b.event("failure_injected", scenario=sc.name, part=sc.key, start_frame=start, **info)
    b.label(phase="failure")
    sc.react(int(rng.integers(*f["reaction_frames"])))
    b.label(phase="detect")
    ftype = sc.detect()
    b.event("failure_detected" if ftype else "failure_not_present", failure_type=ftype,
            truth=observe.truth(b, names), observable=observe.observable(ctx.task, b.kin, b, names))
    if not ftype:                                                  # the injected cause did not lead to a failure
        b.label(phase="normal", failure="")
        return b.i_src + 1, False
    b.label(failure=ftype)
    b.hold(4)
    if ftype.endswith("off_table"):                                # out of reach: cannot continue safely
        b.event("recovery_decision", decision="stop_task", reason="object off the table, out of reach")
        return terminal(ctx, ftype)
    if hasattr(sc, "reachable") and sc.g.obj in b.free and not sc.unmoved():
        ok_r, err = sc.reachable()
        if not ok_r:                                               # lies where no grip can reach it
            ftype = f"{sc.g.key}_out_of_reach"
            b.label(failure=ftype)
            b.event("recovery_decision", decision="stop_task", reason=f"no grip reaches it (IK {err:.3f} m off)")
            return terminal(ctx, ftype)
    ok = False
    max_att = f["retry"]["max_attempts"]
    for attempt in range(1, max_att + 1):
        strat = sc.strategy(attempt)
        fail_again = unrecoverable == "attempts_exhausted" or (      # only the 1st try may fail; the 2nd always
            attempt == 1 and rng.random() < f["retry"]["retry_fail_prob"])   # works (user rule 2026-11-17)
        b.label(phase="recovery", recovery=strat, attempt=attempt)
        b.event("recovery_decision", decision=strat, attempt=attempt)
        sc.recover(attempt, fail_again)
        b.label(phase="verify")
        ok = bool(sc.verify())
        b.event("recovery_verified", strategy=strat, attempt=attempt, success=ok,
                truth=observe.truth(b, names))
        if ok:
            break
        b.label(phase="detect")
        again = sc.detect()
        b.event("failure_detected", failure_type=again or ftype, attempt=attempt, reassessed=True)
        b.hold(4)
    if not ok:
        b.event("recovery_decision", decision="stop_task", reason=f"{max_att} attempts failed")
        return terminal(ctx, ftype)
    r = int(min(max(sc.resume, 0), b.s.T - 1))
    b.label(phase="resume")
    b.i_src = r
    b.blend_to(b.s.q[r], frames=None)
    handover(ctx, r)
    b.event("task_resumed", resume_demo_frame=r)
    b.label(phase="normal", failure="", recovery="", attempt=0)
    return r, False


def handover(ctx, r, pos=0.02, rot_deg=12.0):
    """back on the demo at frame r: an object the recovery holds / put down where the demo has it at r goes back to the
    demo (its later moves - stuck to the carton, carried with it - are the demo's). The small difference is logged."""
    b = ctx.b
    for name in list(b.held) + list(b.free):
        D = b.s.objects[name][r]
        P = b.obj_pose(name)
        e = float(np.linalg.norm(P[:3, 3] - D[:3, 3]))
        a = angle_deg(P[:3, :3], D[:3, :3])
        if e < pos and a < rot_deg:
            b.held.pop(name, None)
            b.free.pop(name, None)
            b.give_back(name)
            b.event("handover_to_demo", object=name, pos_diff_m=round(e, 4), rot_diff_deg=round(a, 2))
        else:
            b.event("kept_off_demo", object=name, pos_diff_m=round(e, 4), rot_diff_deg=round(a, 2))


def terminal(ctx, ftype):
    """give up safely: open the hands, put down what is held, go to the demo's home pose."""
    b = ctx.b
    b.label(phase="terminal", recovery="stop_and_go_home")
    for h in b.hands:
        b.grip_on(h, False)
    for name in list(b.held):
        b.release(name)
    b.blend_to(b.s.q[-1])
    b.event("terminal_failure", failure_type=ftype)
    return b.s.T, True


def run(task, fcfg, src, kin, seed, category=None):
    rng = np.random.default_rng(seed)
    b = Builder(src, task, kin, rng, fcfg["tolerances"])
    grasps = {k: Grasp(k, v, src, kin, task["hands"]) for k, v in task["grasps"].items()}
    ctx = S.Ctx(b, task, fcfg, grasps, rng)
    if not hasattr(kin, "clear"):                                  # arm-to-arm gap from the collision hulls (built once)
        kin.clear = Clearance(task["source"]["urdf"], kin)
    ctx.clear = b.clear = kin.clear
    category = category or pick_category(fcfg, rng)
    chosen, pool = plan(ctx, category)
    unrec = [None] * len(chosen)
    if category == "unrecoverable" and chosen:
        opts = list(fcfg.get("unrecoverable") or ["attempts_exhausted"])
        if not isinstance(chosen[0], S.DropTransport):
            opts = [o for o in opts if o != "off_table"] or ["attempts_exhausted"]
        unrec[0] = opts[int(rng.integers(len(opts)))]
    b.event("episode_start", seed=int(seed), category=category, source=src.kind,
            scenarios=[s.name + ":" + s.key for s in chosen], available=[s.name for s, _ in pool])
    cur, term = 0, False
    for sc, u in zip(chosen, unrec):
        f0 = sc.frame()
        if f0 < cur:
            continue
        b.copy(cur, f0)
        cur, term = handle(ctx, sc, u)
        if term:
            break
    if not term:
        b.copy(cur, src.T)
    b.event("episode_end", terminal=term)
    return b, {"seed": int(seed), "category": category, "terminal": term,
               "scenarios": [{"name": s.name, "part": s.key, "unrecoverable": u} for s, u in zip(chosen, unrec)]}
