"""The failure scenarios: how each failure is caused (inject), how it is noticed (detect, from the scene's actual state),
which recovery is chosen per attempt (strategy) and how it is done (recover) and checked (verify).

Recovery motion is IK (kin.py) to where the object actually is now, never to where the demo had it.
"""
from __future__ import annotations

import numpy as np

from . import detect as D
from .kin import angle_deg, interp_pose


def yaw_only(P):
    """the pose with its tilt removed (kept: position and heading)."""
    M = P.copy()
    x = P[:3, 0].copy()
    x[2] = 0
    x /= np.linalg.norm(x) + 1e-12
    M[:3, :3] = np.c_[x, np.cross([0, 0, 1], x), [0, 0, 1]]
    return M


def rot_z(deg):
    a = np.radians(deg)
    M = np.eye(4)
    M[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return M


def rot_axis(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    M = np.eye(4)
    M[:3, :3] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    return M


def smooth_(t):
    return t * t * (3 - 2 * t)


def safe_offset(ctx, hand, a, e, make, tries=16):
    """a failure's hand offset (make() draws one) that keeps the two arms >= min_arm_gap_m apart all along the demo
    frames a..e it is applied to (or as far apart as the demo itself is there). Returns (offset, gap m)."""
    b, clear = ctx.b, ctx.clear
    hc = b.hands[hand]
    need = ctx.f["tolerances"].get("min_arm_gap_m", 0.025)
    frames = list(range(a, e, 4))
    demo_gap = min(clear.gap(b.s.q[i], b.s.robot_world[i]) for i in frames)
    need = min(need, demo_gap - 0.003)
    best = None
    for _ in range(tries):
        Off = make()
        q = b.q.copy()
        g = 1.0
        for i in frames:
            q0 = b.s.q[i].copy()
            q0[b.s.jidx(hc["arm"])] = q[b.s.jidx(hc["arm"])]
            T = b.kin.link(hc["link"], b.s.q[i], b.s.robot_world[i])
            T2 = Off @ T
            T2[:3, 3] = T[:3, 3] + Off[:3, 3]
            q, _, _ = b.kin.ik(hc["link"], hc["arm"], T2, q0, b.s.robot_world[i])
            g = min(g, clear.gap(q, b.s.robot_world[i]))
        if best is None or g > best[1]:
            best = (Off, g)
        if g >= need:
            break
    return best


def rand_dir(rng):
    a = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(a), np.sin(a), 0.0])


class Ctx:
    def __init__(self, b, task, fcfg, grasps, rng):
        self.b, self.task, self.f, self.g, self.rng = b, task, fcfg, grasps, rng
        self.tol = fcfg["tolerances"]


# ---------------------------------------------------------------------------------------------- shared motions
def advance(b):
    """one more demo frame for the robot (used while something else happens to an object)."""
    if b.i_src + 1 < b.s.T:
        b.i_src += 1
    b.q = b.s.q[b.i_src].copy()
    b.relay = int(b.s.relay[b.i_src])
    b.stage = b.stage_of(b.i_src)


def follow_demo(b, a, e, hand_offsets, ramp_frames=12):
    """demo frames a..e-1, with each listed hand's arm moved by IK to the demo's hand pose shifted by an offset
    (4x4, applied in the world: offset @ demo hand pose), faded in over ramp_frames."""
    for k, i in enumerate(range(a, e)):
        b.i_src = i
        b.stage = b.stage_of(i)
        q = b.s.q[i].copy()
        b.relay = int(b.s.relay[i])
        t = min(1.0, (k + 1) / ramp_frames)
        b.q = np.where(np.isin(np.arange(len(q)), sum((b.s.jidx(b.hands[h]["arm"]) for h in hand_offsets), [])), b.q, q)
        for h, Off in hand_offsets.items():
            demo_hand = b.kin.link(b.hands[h]["link"], q, b.base())
            tgt = interp_pose(np.eye(4), Off, t) @ demo_hand
            tgt[:3, 3] = demo_hand[:3, 3] + t * Off[:3, 3]
            b.q, _, _ = b.kin.ik(b.hands[h]["link"], b.hands[h]["arm"], tgt, b.q, b.base())
        b.push()


def pair_other(b, g, h, ref):
    """user 2026-11-18: the other hand on the strip's other corner, as a person would: it keeps the demo's own
    hand-to-hand placement (both hands carrying the strip at demo frame ref), so wherever this hand grips the strip,
    the other hand's fingers sit on its other corner (not a mirror through the robot's middle, which misses when the
    strip lies off centre)."""
    other = "right" if h == "left" else "left"
    Hh = b.kin.link(b.hands[h]["link"], b.s.q[ref], b.s.robot_world[ref])
    Ho = b.kin.link(b.hands[other]["link"], b.s.q[ref], b.s.robot_world[ref])
    D = np.linalg.inv(Hh) @ Ho
    return other, {other: (lambda: b.hand_pose(h) @ D, np.eye(4))}


def regrasp(ctx, g, fail_again=False, level=False, lift=0.06, carry_to=None):
    """go back to the object where it IS now, grip it (all of g's hands), lift it, and (carry_to) bring the hand to the
    demo's hand pose at that demo frame. level: grip it as if it lay flat (alternative when tilted)."""
    b = ctx.b
    for h in g.hands:
        b.grip_on(h, False)
        b.fingers(h, g.q_open, 8)
    mirror = None
    if g.c.get("mirror_other") and len(g.hands) == 1:            # the other hand moves as the mirror image of this one
        other, mirror = pair_other(b, g, g.hands[0], g.release - 16)   # its own corner of the strip
        b.reach(other, mirror[other][0]())                        # first to the mirror pose, smoothly
    P = b.obj_pose(g.obj)                                         # re-observe
    Pg = yaw_only(P) if level else P
    h = g.hands[0]
    seeds = [g.q_closed, g.q_open]
    if g.c.get("symmetric_yaw", True):                            # a box looks the same turned 180 deg: grip it from the
        alt = Pg @ rot_z(180)                                     # side the arm can actually reach (IK says)
        score = lambda X: b.solve(h, X @ np.linalg.inv(g.rel[h]), [b.q, *seeds])   # noqa: E731
        s0 = score(Pg)
        if s0[1] > 0.01 and score(alt)[1] < s0[1]:                # only when the normal grip cannot reach it
            Pg = alt
    worst = 0.0
    for h in g.hands:
        Tg = Pg @ np.linalg.inv(g.rel[h])
        above = Tg.copy()                                         # come down onto it from straight above
        above[2, 3] += 0.08
        b.goto(h, above, seeds, others=mirror)
        if fail_again:                                            # this attempt misses too
            Tg = Tg.copy()
            Tg[:3, 3] += rand_dir(ctx.rng) * ctx.rng.uniform(0.02, 0.035)
        worst = max(worst, b.goto(h, Tg, seeds, frames=14, others=mirror))
    b.event("regrasp_reach", object=g.obj, worst_pos_err_m=round(float(worst), 4))
    for h in g.hands:
        b.fingers(h, g.q_closed, 8)
        b.grip_on(h, True)
    if mirror:                                                    # the other hand's fingers as this hand's
        oh = next(iter(mirror))
        qf = b.s.q[g.release - 16].copy()                         # its own fingers as the demo has them while both hands
                                                                  # carry the strip (copying the other hand's values bends
                                                                  # the thumb the wrong way)
        fi = b.s.jidx([n for n in b.s.joints if n.startswith(b.hands[oh]["fingers"])])
        b.fingers(oh, qf, max(8, int(np.ceil(1.6 * np.abs(qf[fi] - b.q[fi]).max() / 0.15))))   # no finger jumps
    if not fail_again and worst < 0.015:
        b.attach(g.obj, g.hands)
    b.hold(3)
    h0 = g.hands[0]
    others = {h: (lambda: b.obj_pose(g.obj), b.held.get(g.obj, {}).get(h, g.rel[h])) for h in g.hands[1:]} \
        if g.obj in b.held else None
    if mirror:
        others = {**(others or {}), **mirror}
    up = b.hand_pose(h0)
    up[2, 3] += lift
    b.reach(h0, up, others=others)
    if carry_to is not None:
        b.reach(h0, b.kin.link(b.hands[h0]["link"], b.s.q[carry_to], b.s.robot_world[carry_to]), others=others)
    return worst


def place_at(ctx, g, target):
    """put the held object down at target (from above), open, back off."""
    b = ctx.b
    h0 = g.hands[0]
    others = {h: (lambda: b.obj_pose(g.obj), b.held[g.obj][h]) for h in g.hands[1:]} if g.obj in b.held else None
    rel = b.held[g.obj][h0] if g.obj in b.held else g.rel[h0]
    above = target.copy()
    above[2, 3] += 0.06
    b.reach(h0, above @ np.linalg.inv(rel), others=others)
    b.reach(h0, target @ np.linalg.inv(rel), frames=16, others=others)
    for h in g.hands:
        b.grip_on(h, False)
    b.release(g.obj)
    if g.obj in b.free:
        b.free[g.obj] = target.copy()
    for h in g.hands:
        b.fingers(h, g.q_open, 8)
    b.reach(h0, b.obj_pose(g.obj) @ g.pre[h0], frames=14)


def _footprint_hits(g, L, boxes, margin=0.01):
    """does the object's footprint at pose L touch any keep-out box [[x0, x1], [y0, y1]] (world)?"""
    hx, hy = g.c.get("size_half", [0.05, 0.05, 0])[:2]
    cxy = np.array(g.c.get("size_centre", [0.0, 0.0, 0.0]))[:2]
    pts = [(L @ np.r_[cxy[0] + sx * hx, cxy[1] + sy * hy, 0, 1])[:2] for sx in (-1, 0, 1) for sy in (-1, 0, 1)]
    for (x0, x1), (y0, y1) in boxes:
        for x, y in pts:
            if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin:
                return True
    return False


def landing_ok(ctx, g, L):
    """a landing the robot can see in full (head camera, now, not hidden by an arm) and not on a keep-out area."""
    from .observe import seen
    if _footprint_hits(g, L, g.c.get("keep_out_xy", []), g.c.get("keep_out_margin_m", 0.01)):
        return False
    if g.c.get("keep_in_xy"):                                     # the whole footprint inside the allowed area
        (x0, x1), (y0, y1) = g.c["keep_in_xy"]
        hx, hy = g.c.get("size_half", [0.05, 0.05, 0])[:2]
        c0 = g.c.get("size_centre", [0.0, 0.0, 0.0])
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = (L @ np.r_[c0[0] + sx * hx, c0[1] + sy * hy, 0, 1])[:2]
                if not (x0 <= x <= x1 and y0 <= y <= y1):
                    return False
    b = ctx.b
    if not g.c.get("size_half"):
        return True
    old = b.free.get(g.obj)
    b.free[g.obj] = L
    fr = seen(ctx.task, b.kin, ctx.clear, b, g.obj, g.c["size_half"], centre=g.c.get("size_centre", (0, 0, 0)),
              px=0.0)                                             # in the picture; the hand above may hide it for now:
                                                                  # the camera rule waits until the robot really sees it
    if old is None:
        b.free.pop(g.obj, None)
    else:
        b.free[g.obj] = old
    return fr >= 0.999


def landing_pose(ctx, g, P, scatter, yaw_change, off_table=False, tries=60):
    """landing_pose_once, drawn again until it is fully in the head camera's view and off the keep-out areas (the
    tape stand for the phone): every failure happens where the robot can see it (user rule 2026-11-17)."""
    L = None
    for k in range(tries):
        L = landing_pose_once(ctx, g, P, scatter * (1 + k / 10), yaw_change, off_table)
        if off_table or landing_ok(ctx, g, L):
            return L
    raise RuntimeError(f"{g.obj}: no landing in view / off the keep-out areas after {tries} draws")


def landing_pose_once(ctx, g, P, scatter, yaw_change, off_table=False):
    """where a dropped object comes to rest: under where it fell (+ scatter), flat, turned; in the target (tray) when it
    falls near it; on the floor when off_table."""
    rng, task = ctx.rng, ctx.task
    L = yaw_only(P) @ rot_z(rng.uniform(*yaw_change))
    L[:3, 3] = P[:3, 3] + rand_dir(rng) * rng.uniform(0, scatter)
    if off_table:
        (x0, x1), (y0, y1) = task["table_xy"]
        L[1, 3] = (y0 - 0.15) if L[1, 3] < 0 else (y1 + 0.15)
        L[2, 3] = task["table_z"] - 0.75
        return L
    if g.c.get("lands_in_target") and np.linalg.norm(L[:2, 3] - g.target[:2, 3]) < 0.06:   # into the tray, not flat
        L[2, 3] = g.target[2, 3] + 0.01
        L = L @ rot_axis([1, 0, 0], rng.uniform(5, 20))
    else:
        L[2, 3] = task["table_z"] + g.c.get("on_table_z", g.rest_z - task["table_z"])   # lying on the table
    return L


# ---------------------------------------------------------------------------------------------- scenarios
class Scenario:
    name = ""
    needs_bones = False

    def __init__(self, ctx, cfg, key):
        self.ctx, self.cfg, self.key = ctx, cfg, key
        self.g = ctx.g.get(key)

    def available(self):
        return self.g is not None and self.g.available

    def frame(self):                       # demo frame the failure starts at
        raise NotImplementedError

    def end(self):                         # demo frame the episode continues from after a recovery (upper bound)
        return self.frame() + 60

    def inject(self, unrecoverable):
        raise NotImplementedError

    # ---- the camera rule: the robot does not know yet, it carries on; it notices once its head camera really sees the
    # failed object (in view, not hidden by a hand / arm) for a few frames - at least n frames (5-10), at most max_frames
    def step_unaware(self, k):
        """one more frame of the original motion while the robot does not know."""
        advance(self.ctx.b)
        self.ctx.b.push()

    def deadline(self):
        return self.ctx.b.i_src + self.ctx.f["notice"]["max_frames"]

    def watch(self):
        """(object, half size, centre) the head camera has to see to notice this failure."""
        c, g = self.ctx, self.g
        if g is not None and g.c.get("size_half"):
            return g.obj, g.c["size_half"], g.c.get("size_centre", (0.0, 0.0, 0.0))
        box = c.task["carton_box_local"]                        # the carton (its flaps, lid, or the carton itself)
        lo, hi = np.array(box["min"]), np.array(box["max"])
        return c.task["holder"]["object"], tuple((hi - lo) / 2), tuple((hi + lo) / 2)

    def react(self, n):
        from .observe import seen
        c, b = self.ctx, self.ctx.b
        nc = c.f["notice"]
        obj, half, centre = self.watch()
        last = min(self.deadline(), b.s.T - 1)
        if not self.cfg.get("camera_rule"):                     # visible at once (cases 1-3, 7-8, 10-12): 5-10 frames
            for k in range(1, n + 1):
                if b.i_src >= last:
                    break
                self.step_unaware(k)
            b.event("noticed_after_frames", frames=n)
            return
        run, k, fr = 0, 0, 0.0
        while b.i_src < last:
            k += 1
            self.step_unaware(k)
            fr = seen(c.task, b.kin, c.clear, b, obj, half, centre=centre)
            run = run + 1 if fr >= nc["seen_fraction"] else 0
            if run >= nc["seen_frames"] and k >= n:
                break
        b.event("seen_by_head_camera", object=obj, fraction=round(fr, 2), after_frames=k)

    def detect(self):
        raise NotImplementedError

    def strategy(self, attempt):
        raise NotImplementedError

    def recover(self, attempt, fail_again):
        raise NotImplementedError

    def verify(self):
        raise NotImplementedError


class PickMiss(Scenario):
    """fingers close beside the object (hand misses by a few cm): it stays where it is while the hand lifts."""
    name = "pick_miss"

    def frame(self):
        return self.g.grip - self.g.approach

    def inject(self, unrecoverable):
        c, b, g = self.ctx, self.ctx.b, self.g
        h = g.hands[int(c.rng.integers(len(g.hands)))]

        def make():
            O = np.eye(4)
            O[:3, 3] = rand_dir(c.rng) * c.rng.uniform(*self.cfg.get("miss_offset_m", [0.025, 0.06]))
            return O
        Off, self.gap = safe_offset(c, h, self.frame(), g.grip + 12, make)
        b.free[g.obj] = b.obj_pose(g.obj)
        b.take(g.obj)
        self.offsets = {h: Off}
        follow_demo(b, self.frame(), g.grip + 12, self.offsets)
        return {"hand": h, "miss_m": round(float(np.linalg.norm(Off[:3, 3])), 3), "arm_gap_mm": round(self.gap * 1000, 1)}

    def step_unaware(self, k):
        b = self.ctx.b
        follow_demo(b, b.i_src + 1, b.i_src + 2, self.offsets, ramp_frames=1)

    def deadline(self):
        return min(Scenario.deadline(self), self.g.release - 5)

    def detect(self):
        return D.classify_hold(self.ctx.task, self.ctx.b, self.g, self.ctx.tol, "held")

    def reachable(self, limit=0.03):
        """can a hand still grip it where it lies (IK, both sides of a symmetric object)? (ok, best error m)"""
        b, g = self.ctx.b, self.g
        P = b.obj_pose(g.obj)
        h = g.hands[0]
        cands = [P, P @ rot_z(180)] if g.c.get("symmetric_yaw", True) else [P]
        err = min(b.solve(h, X @ np.linalg.inv(g.rel[h]), [b.q, g.q_closed, g.q_open])[1] for X in cands)
        return err < limit, err

    def unmoved(self):
        """the object still lies exactly where the demo picks it from (a miss that did not disturb it)."""
        b, g = self.ctx.b, self.g
        return b.close(b.obj_pose(g.obj), g.start_pose, 0.003, 1.0)

    def strategy(self, attempt):
        if self.unmoved():
            return ["retry_pick_motion", "retry_pick_motion_slow", "regrasp_at_observed_pose"][min(attempt - 1, 2)]
        return ["regrasp_at_observed_pose", "regrasp_adjusted", "regrasp_adjusted"][min(attempt - 1, 2)]

    def recover(self, attempt, fail_again):
        b, g = self.ctx.b, self.g
        if self.unmoved() and attempt < 3:
            # the object is where it was: open the hand, go back to where the approach started, and do the pick again
            # (the demo's own approach and grip, which works when nothing is off)
            for h in g.hands:
                b.grip_on(h, False)
                b.fingers(h, g.q_open, 8)
            a = g.grip - g.c.get("hover", 15)                     # hovering just above it, as the demo does
            for h in g.hands:
                up = b.hand_pose(h)
                up[2, 3] += 0.05                                  # lift clear first
                b.reach(h, up)
            for h in g.hands:                                     # straight hand path to the demo's hover pose
                b.goto(h, b.kin.link(b.hands[h]["link"], b.s.q[a], b.s.robot_world[a]), [b.s.q[a]])
            b.i_src = a
            b.blend_to(b.s.q[a])                                  # same hand pose: only the elbow settles
            self.resume = max(g.grip + 12, g.lift_frame + 4)
            if fail_again:                                       # misses again
                follow_demo(b, a, self.resume, self.offsets)
                return
            b.free.pop(g.obj, None)
            b.give_back(g.obj)
            slow = 2 if attempt > 1 else 1
            for i in range(a, self.resume):
                for _ in range(slow):
                    b.i_src = i
                    b.q = b.s.q[i].copy()
                    b.relay = int(b.s.relay[i])
                    b.push()
            return
        self.resume = b.i_src
        regrasp(self.ctx, g, fail_again, level=attempt > 1, carry_to=self.resume)

    def verify(self):
        b, g = self.ctx.b, self.g
        if g.obj not in b.held and g.obj not in b.free:          # on the demo again: it has left where it lay, with the
            P = b.obj_pose(g.obj)                                 # hand (a peeled strip or a tipped carton is not rigid in
            moved = np.linalg.norm(P[:3, 3] - g.start_pose[:3, 3]) > 0.003 or \
                D.angle_deg(P[:3, :3], g.start_pose[:3, :3]) > 2   # the hand yet, so not the mid-carry offset)
            return bool(moved and b.close(P, b.s.objects[g.obj][b.i_src], 0.003, 1.0))
        up = b.s.objects[g.obj][b.i_src][2, 3] > g.rest_z + 0.01    # "lifted" only where the demo has it lifted
        return D.is_held(b, g, self.ctx.tol, lifted_from=g.rest_z if up else None)


class VacuumNoLift(PickMiss):
    """the hand comes in and grips as in the demo, the vacuum switches ON, but the object does not come up: the hand
    lifts away empty and the object stays where it was. Recovery: the hand goes back down the demo's own path (vacuum
    off), and grips and lifts again exactly as the demo does (user 2026-11-17: carton flip and carton out)."""
    name = "vacuum_no_lift"

    def frame(self):
        return self.g.grip - self.g.approach

    def _empty_demo(self, a, e):
        """demo frames a..e-1, robot and IO as the demo, the object left where it is."""
        b = self.ctx.b
        for i in range(a, e):
            b.i_src = i
            b.stage = b.stage_of(i)
            b.q = b.s.q[i].copy()
            b.relay = int(b.s.relay[i])
            b.push()

    def inject(self, unrecoverable):
        b, g = self.ctx.b, self.g
        b.free[g.obj] = b.obj_pose(g.obj)
        b.take(g.obj)
        self.offsets = {}
        e = max(g.grip + 12, g.lift_frame + 6)                    # the hands go up; the object stays down
        self._empty_demo(self.frame(), e)
        return {"lifted_away_from": int(g.grip)}

    def step_unaware(self, k):
        b = self.ctx.b
        self._empty_demo(b.i_src + 1, b.i_src + 2)

    def strategy(self, attempt):
        return ["retry_pick_motion", "retry_pick_motion_slow"][min(attempt - 1, 1)]

    def recover(self, attempt, fail_again):
        b, g = self.ctx.b, self.g
        a = g.grip - 2                                            # back down onto it the demo's way and lift from there:
        for h in g.hands:                                         # one touch, one lift (no going away and back in)
            b.grip_on(h, False)                                   # vacuum off
        for i in range(b.i_src, a - 1, -1):                       # the demo's path back down and up to the hover
            b.i_src = i
            b.stage = b.stage_of(i)
            b.q = b.s.q[i].copy()
            b.push()
        b.hold(6)
        self.resume = max(g.grip + 12, g.lift_frame + 6)
        if fail_again:                                            # vacuum on, the object stays down again
            self._empty_demo(a, self.resume)
            return
        b.free.pop(g.obj, None)
        b.give_back(g.obj)
        slow = 2 if attempt > 1 else 1                            # carefully: the demo's pick, slower on a 2nd try
        for i in range(a, self.resume):
            for _ in range(slow):
                b.i_src = i
                b.stage = b.stage_of(i)
                b.q = b.s.q[i].copy()
                b.relay = int(b.s.relay[i])
                b.push()


class PickSlip(PickMiss):
    """gripped and lifted, then it slips out and falls back to the table."""
    name = "pick_slip"

    def frame(self):
        return self.g.grip

    def inject(self, unrecoverable):
        c, b, g = self.ctx, self.ctx.b, self.g
        k = int(c.rng.integers(*self.cfg.get("slip_after", [4, 12])))
        b.copy(g.grip, g.grip + k)
        L = landing_pose(c, g, g.start_pose, 0.02, (-20, 20))
        b.label(phase="failure")
        b.detach_fall(g.obj, L, keep_moving=lambda _: advance(b))
        return {"slip_after": k}

    def step_unaware(self, k):
        Scenario.step_unaware(self, k)


class DropTransport(PickMiss):
    """falls out of the hand while carried; lands on the table, in the tray, or (unrecoverable) off the table."""
    name = "drop_transport"

    def frame(self):
        lo, hi = self.cfg.get("window", [self.g.grip + 20, self.g.release - 20])
        lo, hi = max(lo, self.g.grip + 8), min(hi, self.g.release - 8)
        if not hasattr(self, "_f"):
            self._f = int(self.ctx.rng.integers(lo, max(hi, lo + 1)))
        return self._f

    def inject(self, unrecoverable):
        c, b, g = self.ctx, self.ctx.b, self.g
        off = unrecoverable == "off_table"
        L = landing_pose(c, g, b.obj_pose(g.obj), self.cfg.get("scatter_m", 0.04),
                         self.cfg.get("yaw_change_deg", [-60, 60]), off_table=off)
        b.detach_fall(g.obj, L, keep_moving=lambda _: advance(b))
        return {"landing": [round(float(x), 3) for x in L[:3, 3]], "off_table": off}

    def step_unaware(self, k):
        Scenario.step_unaware(self, k)

    def deadline(self):                    # it has left the hand: the release frame does not limit the noticing
        return Scenario.deadline(self)

    def strategy(self, attempt):
        return ["locate_and_regrasp", "locate_and_regrasp_level", "locate_and_regrasp_level"][min(attempt - 1, 2)]


class Misplaced(Scenario):
    """put down beside / across the target, or tilted (leaning on a wall)."""
    name = "misplaced"

    def frame(self):
        return self.g.release - 25

    def end(self):
        return self.g.release + 60

    def inject(self, unrecoverable):
        """the hand puts it down a few cm off (shifted, slightly turned - the wrist keeps the demo's orientation); it
        lands there, leaning on the tray's edge when it lands on it (tilt is the phone's, not the hand's)."""
        c, b, g = self.ctx, self.ctx.b, self.g
        b.attach(g.obj, g.hands)
        h = g.hands[0]
        tilt = c.rng.uniform(*self.cfg.get("tilt_deg", [0, 18]))

        on_carton = g.c.get("land_on_carton")                     # user 2026-11-18: the strip never off the carton
        Wc = b.obj_pose(c.task["holder"]["object"])
        bx = c.task["carton_box_local"]
        lo, hi = np.array(bx["min"]), np.array(bx["max"])
        P0 = b.s.objects[g.obj][g.release]
        hs, ctr = np.array(g.c.get("size_half", [0, 0, 0])), np.array(g.c.get("size_centre", [0, 0, 0]))

        def on_top(O):
            P = P0.copy()
            P[:3, :3] = O[:3, :3] @ P[:3, :3]
            P[:3, 3] += O[:3, 3]
            C = np.array([[ctr[0] + sx * hs[0], ctr[1] + sy * hs[1], 0, 1] for sx in (-1, 1) for sy in (-1, 1)])
            L = (np.linalg.inv(Wc) @ (P @ C.T)).T[:, :3]
            m = 0.005
            return bool(np.all(L[:, 0] > lo[0] + m) and np.all(L[:, 0] < hi[0] - m)
                        and np.all(L[:, 1] > lo[1] + m) and np.all(L[:, 1] < hi[1] - m))

        def make():
            for _ in range(200):
                O = rot_z(c.rng.uniform(*self.cfg.get("yaw_deg", [-10, 10])))
                O[:3, 3] = rand_dir(c.rng) * c.rng.uniform(*self.cfg.get("offset_m", [0.02, 0.06]))
                if not on_carton or on_top(O):
                    return O
            O = np.eye(4)                                         # nothing fits: a small turn in place
            O[:3, :3] = rot_z(5.0)[:3, :3]
            return O
        Off, self.gap = safe_offset(c, h, self.frame(), g.release + 12, make)
        self.offsets = {h: Off}
        follow_demo(b, self.frame(), g.release + 1, self.offsets)
        P = b.obj_pose(g.obj)
        L = P @ rot_axis([1, 0, 0], tilt)
        L[2, 3] = max(g.target[2, 3] + np.sin(np.radians(tilt)) * 0.04, g.rest_z)
        b.label(phase="failure")
        b.detach_fall(g.obj, L)
        return {"offset_m": round(float(np.linalg.norm(Off[:3, 3])), 3), "tilt_deg": round(float(tilt), 1),
                "arm_gap_mm": round(self.gap * 1000, 1)}

    def step_unaware(self, k):
        """the next steps (the demo's retreat, getting ready to close the lid), the hand's offset fading into the
        demo's path."""
        b = self.ctx.b
        h, Off = next(iter(self.offsets.items()))
        s = max(0.0, 1 - k / 12)
        O = interp_pose(np.eye(4), Off, s)
        O[:3, 3] = Off[:3, 3] * s
        follow_demo(b, b.i_src + 1, b.i_src + 2, {h: O}, ramp_frames=1)

    def deadline(self):
        return min(self.ctx.b.i_src + self.ctx.f["notice"]["max_frames"],
                   self.g.release + self.cfg.get("max_wait", 50))     # before the lid is closed on it

    def detect(self):
        return D.classify_hold(self.ctx.task, self.ctx.b, self.g, self.ctx.tol, "placed")

    def strategy(self, attempt):
        return ["pick_and_replace", "pick_level_and_replace", "pick_level_and_replace"][min(attempt - 1, 2)]

    def recover(self, attempt, fail_again):
        """as the demo places it: grip it the way the demo holds it when it puts it down (palm down, the arm in the
        demo's posture over the tray), lift, go to the demo's hover pose above the tray, then the demo's own placing
        motion (down, release, away)."""
        c, b, g = self.ctx, self.ctx.b, self.g
        s = b.s
        h = g.hands[0]
        hc = b.hands[h]
        rf = g.release - 2                                        # the demo's hand on the phone as it puts it down
        hover = g.release - g.c.get("place_hover", 20)            # the demo's hand above the tray before that
        O = s.objects[g.obj]
        rel = np.linalg.inv(b.kin.link(hc["link"], s.q[rf], s.robot_world[rf])) @ O[rf]
        seeds = [s.q[rf], s.q[hover]]
        for hh in g.hands:
            b.grip_on(hh, False)
            b.fingers(hh, g.q_open, 8)
        # user 2026-11-18: the tape is set right with BOTH hands' fingers, as the demo pastes it: the other hand moves
        # as the mirror image of this one all along (its fingers as in the demo's own placing)
        mo = None
        if g.c.get("mirror_other") and len(g.hands) == 1:
            other, mo = pair_other(b, g, h, rf)                    # its own corner of the strip, as the demo
            b.reach(other, mo[other][0]())
            fo = [i for i, n in enumerate(s.joints) if n.startswith(b.hands[other]["fingers"])]
        up = b.hand_pose(h)
        up[2, 3] += 0.05                                          # lift clear of it
        b.reach(h, up, others=mo)
        P = b.obj_pose(g.obj)                                     # re-observe
        Pg = yaw_only(P) if attempt > 1 else P
        Tg = Pg @ np.linalg.inv(rel)
        above = Tg.copy()
        above[2, 3] += 0.06
        b.goto(h, above, seeds, others=mo)
        if fail_again:                                            # this attempt misses
            Tg = Tg.copy()
            Tg[:3, 3] += rand_dir(c.rng) * c.rng.uniform(0.02, 0.035)
        err = b.goto(h, Tg, seeds, frames=16, others=mo)
        b.event("regrasp_reach", object=g.obj, worst_pos_err_m=round(float(err), 4))
        fi = [i for i, n in enumerate(s.joints) if n.startswith(hc["fingers"])]
        q_hold = b.q.copy()
        q_hold[fi] = s.q[rf][fi]
        if mo:
            q_hold[fo] = s.q[rf][fo]                              # the other hand's fingers on it too
        b.fingers(h, q_hold, 8)                                   # the demo's grip on it while placing
        if mo:
            b.fingers(other, q_hold, 6)                           # and the other hand's, as in the demo
        b.grip_on(h, True)
        self.resume = b.i_src
        if fail_again or err > 0.015:
            b.hold(4)
            return
        b.attach(g.obj, [h])
        b.hold(3)
        r0 = b.held[g.obj][h].copy()                              # as it was gripped (maybe leaning)
        # user 2026-11-18: lift it a little straight up (clear of the tray walls), over its place, straight down - never
        # dragged sideways through the tray
        Hp = b.kin.link(hc["link"], s.q[rf], s.robot_world[rf])  # the demo's hand as the phone sits in its place
        clear = Hp[2, 3] + 0.06
        up = b.hand_pose(h)
        up[2, 3] = max(up[2, 3] + 0.03, clear)
        n0 = b.n
        b.reach(h, up, frames=18, speed=0.08, others=mo)
        for k, i in enumerate(range(n0, b.n)):                    # lifting it, the fingers square it against the
            t = smooth_((k + 1) / (b.n - n0))                     # palm: from its lean to the demo's flat hold
            R = interp_pose(r0, rel, t)
            b.held[g.obj][h] = R
            Hh = b.kin.link(hc["link"], np.array(b.out["q"][i]), s.robot_world[b.out["src"][i]])
            b.obj_out[g.obj][i] = Hh @ R
        b.held[g.obj][h] = rel.copy()
        over = Hp.copy()
        over[2, 3] = up[2, 3]
        e1 = b.reach(h, over, frames=24, others=mo)                          # across, high: a straight hand path
        q_hi, _, _ = b.kin.ik(hc["link"], hc["arm"], over, s.q[rf], s.robot_world[rf])   # the demo's elbow, up here
        q_t = b.q.copy()
        q_t[s.jidx(hc["arm"])] = q_hi[s.jidx(hc["arm"])]
        b.blend_to(q_t)                                           # (settles high above the carton, not at it)
        if mo:
            b.reach(h, b.hand_pose(h), frames=8, others=mo)       # the other hand catches up with the mirror pose
        e2 = b.reach(h, Hp, frames=20, others=mo)                            # straight down into place
        b.event("replace_path", lift_z=round(float(up[2, 3]), 3), across=[bool(e1[0]), round(float(e1[1]), 4)],
                down=[bool(e2[0]), round(float(e2[1]), 4)])
        # the demo's own placing from here (the tape is pressed on only during the demo's release, user 2026-11-18): the
        # arm settles into the demo's joints at rf while the hand stays where it is (IK holds it, 0.5 cm checked)
        q_ik = b.q.copy()
        n1 = b.n
        b.blend_to(s.q[rf])
        dev = max(float(np.linalg.norm(b.kin.link(hc["link"], np.array(b.out["q"][i]), s.robot_world[rf])[:3, 3]
                                       - Hp[:3, 3])) for i in range(n1, b.n)) if b.n > n1 else 0.0
        b.event("hand_back_to_demo", hand_dev_mm=round(dev * 1000, 1),
                elbow_change_rad=round(float(np.abs(q_ik - s.q[rf]).max()), 3))
        b.i_src = rf
        b.copy(rf, g.release + 12)                                # the demo's own press / release and retreat
        self.resume = g.release + 12

    def verify(self):
        return D.is_placed(self.ctx.b, self.g, self.ctx.tol)


class BoxFallsOnRelease(Scenario):
    """the carton is in the holder; when the (left) index finger lets go it tips over."""
    name = "falls_on_release"

    def frame(self):
        return self.g.c.get("release_frames", [self.g.release + 5])[0] + 4

    def end(self):
        return self.frame() + 40

    def inject(self, unrecoverable):
        c, b, g = self.ctx, self.ctx.b, self.g
        P = b.obj_pose(g.obj)
        tilt = c.rng.uniform(*self.cfg.get("tilt_deg", [35, 85]))
        box = c.task["carton_box_local"]                          # the carton's real size (task yaml)
        lo, hi = np.array(box["min"]), np.array(box["max"])
        corners = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        W = (P @ corners.T).T[:, :3]
        bottom = W[np.argsort(W[:, 2])[:4]]                       # the face it stands on
        pivot = bottom.mean(0)
        thin = np.argmin(hi - lo)                                 # it falls across its thinnest side (3.6 cm)
        n = P[:3, thin].copy()                                    # that side's direction, horizontal
        n[2] = 0
        n /= np.linalg.norm(n) + 1e-9
        n *= 1 if c.rng.random() < 0.5 else -1                    # forward or backward
        pivot = pivot + n * (hi - lo)[thin] / 2                   # the bottom edge on that side
        axis = np.cross([0, 0, 1], n)
        frames = 12
        self.start = P.copy()
        self.pivot, self.axis, self.tilt = pivot, axis, tilt
        for k in range(1, frames + 1):
            R = rot_axis(axis, tilt * (k / frames) ** 2)
            M = R @ P
            M[:3, 3] = pivot + R[:3, :3] @ (P[:3, 3] - pivot)
            b.free[g.obj] = M
            b.take(g.obj)
            advance(b)
            b.push()
        return {"tilt_deg": round(float(tilt), 1)}

    def detect(self):
        b, g = self.ctx.b, self.g
        e, r = D.placed_error(b.obj_pose(g.obj), self.start)
        return None if r < self.ctx.tol["upright_deg"] else f"{g.key}_fell_in_holder"

    def strategy(self, attempt):
        return ["push_back_upright_left_hand", "push_back_upright_slowly", "push_back_upright_slowly"][min(attempt - 1, 2)]

    def recover(self, attempt, fail_again):
        """the hand that let go (left) comes back onto the carton's top edge, pushes it upright about the bottom edge it
        tipped over, holds it still, and backs off slowly."""
        c, b, g = self.ctx, self.ctx.b, self.g
        rf = g.c.get("release_frames", [g.release + 5])
        self.resume = rf[0]
        h = (g.c.get("release_order") or g.hands)[0]
        F = b.obj_pose(g.obj)
        box = c.task["carton_box_local"]
        lo, hi = np.array(box["min"]), np.array(box["max"])
        corners = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        Wc = (F @ corners.T).T[:, :3]
        top = Wc[np.argsort(-np.linalg.norm(Wc - self.pivot, axis=1))[:4]].mean(0)   # the face furthest from the pivot
        frac = [1.0, 0.7, 0.5][min(attempt - 1, 2)]               # later attempts push lower down (easier to reach)
        T = b.hand_pose(h)
        T[:3, 3] = self.pivot + frac * (top - self.pivot)
        seeds = [b.s.q[rf[0]], b.s.q[g.grip + 10], b.s.q[g.grip - 10]]
        err = b.goto(h, T, seeds)
        b.event("push_contact", hand=h, pos_err_m=round(float(err), 4))
        if err > 0.02:
            return                                                # cannot reach it: this attempt fails
        rel = np.linalg.inv(b.hand_pose(h)) @ F
        hc = b.hands[h]
        end = 0.5 if fail_again else 0.0                          # fail_again: the push stops half way
        N = 24 * (2 if attempt > 1 else 1)
        for k in range(1, N + 1):
            ang = self.tilt * (1 - (1 - end) * smooth_(k / N))
            R = rot_axis(self.axis, ang)
            M = R @ self.start
            M[:3, 3] = self.pivot + R[:3, :3] @ (self.start[:3, 3] - self.pivot)
            q, _, _ = b.kin.ik(hc["link"], hc["arm"], M @ np.linalg.inv(rel), b.q, b.base())
            b.step_to(q)
            b.free[g.obj] = M
            b.push()
        if fail_again:                                            # it was not upright: falls back when let go
            b.hold(4)
            b.free[g.obj] = F
            b.push()
            return
        b.free[g.obj] = self.start.copy()
        b.hold(10)                                                # hold it still
        b.i_src = self.resume
        b.blend_to(b.s.q[self.resume], frames=40)                 # back off slowly

    def step_unaware(self, k):
        Scenario.step_unaware(self, k)

    def verify(self):
        b = self.ctx.b
        b.hold(15)                                              # still standing after 0.5 s without support
        e, r = D.placed_error(b.obj_pose(self.g.obj), self.start)
        return r < self.ctx.tol["upright_deg"] and e < self.ctx.tol["placed_pos_m"]


class PartNotFolded(Scenario):
    """a carton part (side flap, lid, tuck lip) stops part way (or springs back) while the fingers fold it."""
    name = "not_folded"
    needs_bones = True

    def __init__(self, ctx, cfg, key):
        super().__init__(ctx, cfg, key)
        self.part = ctx.task.get("carton_parts", {}).get(key)
        self.g = None

    def available(self):
        p = self.part
        return bool(p and p["bones"] and all((p["armature"], x) in self.ctx.b.s.bones for x in p["bones"]))

    def frame(self):
        return self.part["fold"][0]

    def end(self):
        return self.part["fold"][1] + 40

    def _set(self, frac_of):
        """the part this far folded, ON ITS HINGE (relative to the panel it hangs on, which keeps moving)."""
        b, p = self.ctx.b, self.part
        a, e = p["fold"]
        for x in p["bones"]:
            key = (p["armature"], x)
            b.bone_rel[key] = interp_pose(self._local(key, a), self._local(key, e), frac_of)

    def _local(self, key, i):
        par = self.ctx.b.s.bone_parent.get(key)
        B = self.ctx.b.s.bones[key][i]
        if not par:
            return B
        return np.linalg.inv(self.ctx.b.s.bones[(key[0], par)][i]) @ B

    def inject(self, unrecoverable):
        c, b, p = self.ctx, self.ctx.b, self.part
        a, e = p["fold"]
        self.reach_frac = c.rng.uniform(*self.cfg.get("reach", [0.2, 0.7]))
        for i in range(a, e):
            b.i_src = i
            b.q = b.s.q[i].copy()
            b.relay = int(b.s.relay[i])
            b.stage = b.stage_of(i)
            self._set(min((i - a) / (e - a), self.reach_frac))
            b.push()
        return {"reach": round(float(self.reach_frac), 2)}

    def detect(self):
        err = D.part_error_deg(self.ctx.b, self.part, self.part["fold"][1])
        return None if err < self.ctx.tol["flap_deg"] else f"{self.key}_not_folded"

    def strategy(self, attempt):
        return ["refold_this_part", "refold_this_part_slowly", "refold_this_part_slowly"][min(attempt - 1, 2)]

    def recover(self, attempt, fail_again):
        if len(self.part["bones"]) == 1 and self.part["hand"] in ("left", "right"):
            return self.push_with_finger(attempt, fail_again)      # a side flap: a finger pushes it closed
        # the lid / tuck lip: as a person would - both hands on it where it stopped, LIFT it back up to vertical (the
        # demo's closing played backwards), then CLOSE it again (the demo's closing forwards)
        b, p = self.ctx.b, self.part
        a, e = p["fold"]
        self.resume = b.i_src
        hands = ["right", "left"] if p["hand"] == "both" else [p["hand"]]
        joints = sum(([n for n in b.s.joints if n.startswith(b.hands[h]["fingers"])] + b.hands[h]["arm"] for h in hands), [])
        idx = b.s.jidx(joints)
        frozen = {k: b.bone_local(k) for k in b.s.bones if k[0] == p["armature"] and k[1] not in p["bones"]}
        b.bone_rel.update(frozen)                                  # the rest of the carton stays as it is now, each
                                                                   # part on its own hinge (flaps follow the lid panel)
        cur = getattr(self, "cur_frac", self.reach_frac)           # how far it is closed now
        fr = int(round(a + cur * (e - a)))                         # the demo's moment with the lid at that angle
        slow = 2 if attempt > 1 else 1
        if self.key == "lid":
            self.thumb_lift(fr, a, e, slow, joints)                # the LEFT thumb lifts it, as in the demo's opening
        else:
            b.blend_to(b.s.q[fr], joints=joints)                   # hands onto the part where it stopped
            for i in range(fr, a - 1, -1):                         # lift it back up (demo's closing backwards)
                for _ in range(slow):
                    b.q[idx] = b.s.q[i][idx]
                    self._set((i - a) / (e - a))
                    b.push()
        top = 0.6 if fail_again else 1.0
        if self.key == "lid":
            # the hands go round the open lid the demo's way: raised, then over and down behind it (demo 365-404,
            # the lid pushed up from leaning back to upright), then close it (404-445) - never through the lid
            b.blend_to(b.s.q[self.BEHIND], joints=joints)
            for i in range(self.BEHIND, e + 1):
                for _ in range(slow):
                    b.q[idx] = b.s.q[i][idx]
                    if i < a:
                        for x in p["bones"]:
                            b.bone_rel[(p["armature"], x)] = self._local((p["armature"], x), i)
                    else:
                        self._set(min((i - a) / (e - a), top))
                    b.push()
        else:
            b.blend_to(b.s.q[a], joints=joints)
            b.hold(8)
            for i in range(a, e + 1):                              # close it again
                for _ in range(slow):
                    b.q[idx] = b.s.q[i][idx]
                    self._set(min((i - a) / (e - a), top))
                    b.push()
        self.cur_frac = top
        if not fail_again:
            for x in p["bones"]:
                b.bone_ovr.pop((p["armature"], x), None)
                b.bone_rel.pop((p["armature"], x), None)
        for k in frozen:
            b.bone_rel.pop(k, None)
        b.event("lid_lifted_and_closed", part=self.key, from_closed=round(float(cur), 2), closed_to=top)
        b.blend_to(b.s.q[self.resume], joints=joints)


    # ---- the lid: the LEFT thumb lifts it, exactly as the demo's left hand opens the lid in its first 10 s (OpenLid,
    # frames 185-224: hand comes in, thumb under the front edge from 204, lid upright at ~224). The demo's hand pose is
    # replayed RELATIVE TO THE LID PANEL, so the thumb stays under the edge wherever the lid stopped (user 2026-11-17)
    OPEN_FROM, OPEN_TOUCH, OPEN_UP = 185, 204, 230
    BEHIND = 300                     # demo 10-14 s: the left hand keeps the open lid, the right comes behind it, both close it

    def thumb_lift(self, fr, a, e, slow, joints):
        b, p = self.ctx.b, self.part
        s, A = b.s, p["armature"]
        E3k = (A, p["bones"][0])                                  # the lid panel (task file: the lid's first bone)
        lid_keys = [(A, x) for x in p["bones"]]

        def world_E3_demo(i):
            return s.objects[A][i] @ s.bones[E3k][i]

        def world_E3_now():
            M, k = np.eye(4), E3k
            while k:
                M = b.bone_local(k) @ M
                par = s.bone_parent.get(k)
                k = (A, par) if par else None
            return b.obj_pose(A) @ M

        hc = b.hands["left"]
        Lf = s.jidx([n for n in s.joints if n.startswith(hc["fingers"])])
        rel = {i: np.linalg.inv(world_E3_demo(i)) @ b.kin.link(hc["link"], s.q[i], s.robot_world[i])
               for i in range(self.OPEN_FROM, self.OPEN_UP + 1)}
        ang = lambda i: angle_deg(s.bones[E3k][self.OPEN_TOUCH][:3, :3], s.bones[E3k][i][:3, :3])   # noqa: E731
        span = ang(self.OPEN_UP)
        start = {k: b.bone_local(k).copy() for k in lid_keys}
        up = {k: self._local(k, self.OPEN_UP) for k in lid_keys}   # open = the demo's lid after its opening

        def move_hand(i, frames):
            T0 = b.hand_pose("left")
            T1 = world_E3_now() @ rel[i]
            f0 = b.q[Lf].copy()
            for k in range(1, frames + 1):
                tt = smooth_(k / frames)
                T = interp_pose(T0, T1, tt)
                q, _, _ = b.kin.ik(hc["link"], hc["arm"], T, b.q, b.base())
                q[Lf] = (1 - tt) * f0 + tt * s.q[i][Lf]
                b.step_to(b.kin.clamp(q))
                b.push()

        move_hand(self.OPEN_FROM, 24)                              # left hand to where the demo's starts its opening
        for i in range(self.OPEN_FROM + 1, self.OPEN_UP + 1):     # the demo's opening, frame by frame
            for _ in range(slow):
                if i > self.OPEN_TOUCH:                            # the thumb under the edge turns the lid up
                    f = min(ang(i) / span, 1.0)
                    for k in lid_keys:
                        b.bone_rel[k] = interp_pose(start[k], up[k], f)
                q, _, _ = b.kin.ik(hc["link"], hc["arm"], world_E3_now() @ rel[i], b.q, b.base())
                q[Lf] = s.q[i][Lf]
                b.step_to(b.kin.clamp(q))
                b.push()
        b.hold(8)                                                  # open: the left hand keeps it there (demo 230-300)

    # ---- a person's way: put a finger on the flap's free edge and push it closed (the flap only turns as far as the
    # fingertip pushes it), take the finger away, carry on
    def _flap_contact_local(self):
        """a point on the middle of the flap's free edge, in the flap bone's own frame (from the carton mesh)."""
        import json as _json
        from .export import B2I
        p, s = self.part, self.ctx.b.s
        bone = p["bones"][0]
        Z = np.load(B2I / "blender_export" / (s.scene or "gidora_packing") / "objects_export.npz")
        info = _json.loads(str(Z["info"]))
        k = next(i for i, o in enumerate(info["objects"])               # the carton mesh: the one this bone bends
                 if p["armature"] in o.get("armature", []) and bone in o.get("vertex_groups", []))
        gi = info["objects"][k]["vertex_groups"].index(bone)
        e = p["fold"][1]
        V = Z[f"vw_{k}"][e][Z[f"vg_{k}"] == gi].astype(float)
        Bw = s.objects[p["armature"]][e] @ s.bones[(p["armature"], bone)][e]
        L = (np.linalg.inv(Bw) @ np.c_[V, np.ones(len(V))].T).T[:, :3]
        d = np.linalg.norm(L - 0.0, axis=1)                        # distance from the hinge (bone head)
        edge = L[d >= np.quantile(d, 0.85)]
        return edge.mean(0)

    def push_with_finger(self, attempt, fail_again):
        c, b, p = self.ctx, self.ctx.b, self.part
        s = b.s
        bone = p["bones"][0]
        key = (p["armature"], bone)
        a, e = p["fold"]
        h = p["hand"]
        hc = b.hands[h]
        tip_link = hc["fingers"].replace("_", "_link_") + "53"     # the LITTLE finger pushes it (user's choice
        pts = c.clear.pts[tip_link]                                # 2026-11-17): its fingertip on the middle of the
        tip = pts[np.argmax(np.linalg.norm(pts, axis=1))]          # flap's free edge (RF_link_53 / LF_link_53)
        self.resume = b.i_src
        Lopen = b.bone_local(key).copy()                           # where the flap stopped, on its hinge
        frozen = {k: v[b.i_src] for k, v in s.bones.items() if k[0] == p["armature"] and k != key}
        b.bone_ovr.update(frozen)                                  # the rest of the carton stays as it is now
        b.bone_ovr.pop(key, None)
        par = s.bone_parent.get(key)
        Ppar = b.bone_ovr[(p["armature"], par)] if par else np.eye(4)
        Lshut = self._local(key, e)                                # folded, on its hinge
        Bopen, Bshut = Ppar @ Lopen, Ppar @ Lshut
        Wbox = b.obj_pose(p["armature"])
        cl = np.r_[self._flap_contact_local(), 1.0]
        contact = lambda Bm: (Wbox @ Bm @ cl)[:3]                  # noqa: E731
        c0, c1 = contact(Bopen), contact(Bshut)
        d = (c1 - c0) / (np.linalg.norm(c1 - c0) + 1e-9)          # the way the edge moves when pushed shut
        R0 = b.kin.link(tip_link, b.q, b.base())[:3, :3]          # keep the finger's orientation

        def tip_to(point, frames):
            start = b.kin.link(tip_link, b.q, b.base())
            p0 = start[:3, 3] + start[:3, :3] @ tip
            for k in range(1, frames + 1):
                tt = smooth_((k) / frames)
                want = (1 - tt) * p0 + tt * point
                T = np.eye(4)
                T[:3, :3] = R0
                T[:3, 3] = want - R0 @ tip
                q, _, _ = b.kin.ik(tip_link, hc["arm"], T, b.q, b.base())
                b.step_to(q)
                b.push()

        tip_to(c0 - 0.035 * d + np.array([0, 0, 0.02]), 30)        # finger to just behind the free edge
        tip_to(c0 - 0.004 * d, 12)                                 # touch it
        N = 30 * (2 if attempt > 1 else 1)
        top = 0.5 if fail_again else 1.0                           # fail_again: it slips off half way
        for k in range(1, N + 1):                                  # push: the flap turns with the fingertip
            f = top * smooth_(k / N)
            b.bone_rel[key] = interp_pose(Lopen, Lshut, f)
            Bm = Ppar @ b.bone_rel[key]
            T = np.eye(4)
            T[:3, :3] = R0
            T[:3, 3] = contact(Bm) - 0.004 * d - R0 @ tip
            q, _, _ = b.kin.ik(tip_link, hc["arm"], T, b.q, b.base())
            b.step_to(q)
            b.push()
        b.hold(6)                                                  # hold it there a moment
        tip_to(contact(Ppar @ b.bone_rel[key]) - 0.04 * d + np.array([0, 0, 0.03]), 18)   # finger away
        if fail_again:
            b.bone_rel[key] = Lopen                                # the finger slipped: the flap springs back
        else:
            b.bone_rel.pop(key, None)                              # folded = the demo's folded flap from here on
        for k in frozen:
            b.bone_ovr.pop(k, None)
        b.event("flap_pushed_by_finger", part=self.key, finger=tip_link, reached=round(float(top), 2))

    def verify(self):
        b = self.ctx.b
        b.hold(3)
        return D.part_error_deg(b, self.part, self.part["fold"][1]) < self.ctx.tol["flap_deg"]


# scenario name in failures.yaml -> (class, grasp or carton part key)
REGISTRY = {
    "phone_pick_miss": (PickMiss, "phone"),
    "phone_pick_slip": (PickSlip, "phone"),
    "phone_drop_transport": (DropTransport, "phone"),
    "phone_misplaced": (Misplaced, "phone"),
    "flap_not_folded": (PartNotFolded, "side_flap_left|side_flap_right"),
    "lid_not_closed": (PartNotFolded, "lid|tuck_lip"),
    "tape_not_lifted": (PickMiss, "tape"),
    "tape_drop_transport": (DropTransport, "tape"),
    "tape_misapplied": (Misplaced, "tape"),
    "box_lift_miss": (VacuumNoLift, "box_flip"),        # vacuum on, carton not lifted (user 2026-11-17)
    "box_out_no_lift": (VacuumNoLift, "box_out"),       # vacuum on, carton not lifted out
    "box_falls_on_release": (BoxFallsOnRelease, "box_flip"),
    "box_drop_out": (DropTransport, "box_out"),
}


def make(ctx, name, cfg):
    cls, keys = REGISTRY[name]
    keys = keys.split("|")
    sc = cls(ctx, cfg, keys[int(ctx.rng.integers(len(keys)))])
    sc.name = name
    return sc
