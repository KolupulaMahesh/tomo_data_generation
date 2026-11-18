"""Builds one episode frame by frame: the demo's own frames, plus failure / recovery frames made by IK.

Every output frame stores:
  src      the demo frame the scene (carton, tape, camera, everything not overridden) is taken from
  q        the robot's joints (the demo's, or IK / blends in failure and recovery frames)
  objects  the world pose of every tracked object: the demo's, or "held" (hand pose @ grip offset), or "free"
           (resting / falling where the failure left it)
  relay    the IO byte (grip on / off per hand)
  labels   stage, phase, failure type, recovery strategy, attempt

Physics is off: a dropped object follows a computed fall (gravity, then rests on the table / tray), objects never jump.
"""
from __future__ import annotations

import numpy as np

from .kin import Kin, angle_deg, interp_pose

PHASES = ["normal", "failure", "detect", "recovery", "verify", "resume", "terminal"]
G = 9.81
MAX_STEP = 0.05             # rad per frame for any joint in generated motion (smooth; speed is not a constraint)


def smooth(t):
    return t * t * (3 - 2 * t)


class Builder:
    def __init__(self, src, task, kin: Kin, rng, tol):
        self.s, self.task, self.kin, self.rng, self.tol = src, task, kin, rng, tol
        self.hands = task["hands"]
        self.q = src.q[0].copy()
        self.i_src = 0
        self.held = {}              # obj -> {hand: rel (hand^-1 @ obj)}, first hand carries it
        self.free = {}              # obj -> pose
        self.bone_freeze = {}       # obj (an armature) -> demo frame its bones are frozen at while it is held / free
        self.bone_ovr = {}          # (armature, bone) -> pose, carton parts left part way
        self.carried = {}           # obj of ours -> (frame taken, objects moving with it)
        self.bone_rel = {}          # (armature, bone) -> pose relative to its parent bone (follows the parent)
        self.relay = int(src.relay[0])
        self.stage = task["stages"][0][0]
        self.lab = dict(phase="normal", failure="", recovery="", attempt=0)
        self.out = dict(src=[], q=[], relay=[], stage=[], phase=[], failure=[], recovery=[], attempt=[])
        self.obj_out = {n: [] for n in src.objects}
        self.bone_out = []          # per frame: {(arm, bone): pose} overrides (only non-demo ones)
        self.events = []

    # ------------------------------------------------------------------ state
    @property
    def n(self):
        return len(self.out["q"])

    def base(self, i=None):
        return self.s.robot_world[self.i_src if i is None else i]

    def hand_pose(self, hand, q=None):
        return self.kin.link(self.hands[hand]["link"], self.q if q is None else q, self.base())

    def take(self, name):
        """the generator takes over this object's pose from now on: freeze its bones' shape, and find the objects that
        move rigidly with it in the demo around now (parented, or stuck to it like the tape on the carton, the phone in
        it): they move with it while it is ours."""
        if name in self.bone_freeze:
            return
        f = self.i_src
        self.bone_freeze[name] = f
        O, T = self.s.objects, self.s.T
        win = range(max(f - 5, 0), min(f + 60, T), 3)
        A = O[name]
        mine = set()
        for n, B in O.items():
            if n == name or n in self.s.robot_attached or n in self.held or n in self.free:
                continue
            if self.s.follows.get(n) == name:
                mine.add(n)
                continue
            R = [np.linalg.inv(A[i]) @ B[i] for i in win]
            if max(np.linalg.norm(r[:3, 3] - R[0][:3, 3]) for r in R) < 0.002 and \
                    max(np.linalg.norm(B[i][:3, 3] - B[f][:3, 3]) for i in win) > 0.01:
                mine.add(n)
        self.carried[name] = (f, mine)

    def bone_local(self, key, i=None):
        """a bone's pose relative to its parent bone, at demo frame i (now if None) - or as overridden now."""
        if key in self.bone_rel:
            return self.bone_rel[key]
        i = self.i_src if i is None else i
        par = self.s.bone_parent.get(key)
        B = self.bone_ovr.get(key, self.s.bones[key][i]) if i == self.i_src else self.s.bones[key][i]
        if not par:
            return B
        pk = (key[0], par)
        P = self.bone_ovr.get(pk, self.s.bones[pk][i]) if i == self.i_src else self.s.bones[pk][i]
        return np.linalg.inv(P) @ B

    def give_back(self, name):
        """the demo drives this object (and what moves with it) again."""
        self.bone_freeze.pop(name, None)
        self.carried.pop(name, None)

    def obj_pose(self, name, i=None):
        i = self.i_src if i is None else i
        for root, (f, mine) in self.carried.items():                # moves with an object of ours
            if name in mine and (root in self.held or root in self.free):
                O = self.s.objects
                return self.obj_pose(root, i) @ np.linalg.inv(O[root][f]) @ O[name][f]
        root = self.s.follows.get(name)
        if root and (root in self.held or root in self.free):      # hangs under an object of ours: moves with it
            f = self.bone_freeze.get(root, i)
            O = self.s.objects
            return self.obj_pose(root, i) @ np.linalg.inv(O[root][f]) @ O[name][f]
        if name in self.held:
            hand, rel = next(iter(self.held[name].items()))
            return self.hand_pose(hand) @ rel
        if name in self.free:
            return self.free[name]
        return self.s.objects[name][i]

    def stage_of(self, i):
        st = self.task["stages"][0][0]
        for name, f in self.task["stages"]:
            if i >= f:
                st = name
        return st

    def hand_bits(self, hand):
        return sum(1 << b for b in self.hands[hand]["io_bits"])

    def grip_on(self, hand, on=True):
        self.relay = (self.relay | self.hand_bits(hand)) if on else (self.relay & ~self.hand_bits(hand) & 0xFF)

    def label(self, **kw):
        self.lab.update(kw)

    def event(self, kind, **kw):
        e = dict(kind=kind, frame=self.n, demo_frame=int(self.i_src), stage=self.stage, **kw)
        self.events.append(e)
        return e

    # ------------------------------------------------------------------ writing frames
    def push(self):
        o = self.out
        o["src"].append(self.i_src)
        o["q"].append(self.q.copy())
        o["relay"].append(self.relay)
        o["stage"].append(self.stage)
        for k in ("phase", "failure", "recovery", "attempt"):
            o[k].append(self.lab[k])
        for name in self.obj_out:
            self.obj_out[name].append(self.obj_pose(name))
        b = dict(self.bone_ovr)
        done, i = {}, self.i_src
        moved = {k[0] for k in self.bone_rel} | {k[0] for k in self.bone_ovr}

        def resolve(key):                                         # every part stays on its hinge: parent's pose NOW @
            if key in done:                                       # its own pose on the hinge (parents first)
                return done[key]
            par = self.s.bone_parent.get(key)
            pk = (key[0], par) if par else None
            if key in self.bone_rel:                              # ours, held on its hinge
                done[key] = (resolve(pk) if pk else np.eye(4)) @ self.bone_rel[key]
            elif key in b:                                        # ours, held as a whole
                done[key] = b[key]
            elif pk and pk in self.s.bones:                       # the demo's angle on its hinge, on its parent as it
                Pd = self.s.bones[pk][i]                          # really is now (a flap on a lid panel stopped part way)
                done[key] = resolve(pk) @ np.linalg.inv(Pd) @ self.s.bones[key][i]
            else:
                done[key] = self.s.bones[key][i]
            return done[key]
        for key in self.s.bones:
            if key[0] in moved:
                r = resolve(key)
                if not np.allclose(r, self.s.bones[key][i]):
                    b[key] = r
        for name, f in self.bone_freeze.items():
            for (arm, bone), P in self.s.bones.items():
                if arm == name:
                    b[(arm, bone)] = P[f]
        self.bone_out.append(b)

    def copy(self, a, b):
        """The demo's frames a..b-1 as they are (held / free objects stay ours until the demo agrees again)."""
        for i in range(a, b):
            self.i_src = i
            self.q = self.s.q[i].copy()
            prev = self.relay
            self.relay = int(self.s.relay[i])
            self.stage = self.stage_of(i)
            for hand in self.hands:                                   # the demo opens a hand that holds one of ours
                bits = self.hand_bits(hand)
                if prev & bits and not self.relay & bits:
                    for name in [n for n, h in self.held.items() if hand in h]:
                        self.release(name, hand)
            for name in list(self.free):                              # the demo agrees again: back to the demo's pose
                if name in self.s.objects and self.close(self.free[name], self.s.objects[name][i], 0.002, 1.0):
                    del self.free[name]
                    self.give_back(name)
            for name in list(self.held):
                if name in self.s.objects and self.close(self.obj_pose(name), self.s.objects[name][i], 0.002, 1.0) \
                        and self.demo_holds(name, i):
                    del self.held[name]
                    self.give_back(name)
            self.push()

    def demo_holds(self, name, i):
        g = self.task["grasps"]
        return any(v["object"] == name and v["grip"] <= i < v["release"] for v in g.values())

    def hold(self, frames):
        for _ in range(frames):
            self.push()

    def close(self, A, B, pos, rot_deg):
        return np.linalg.norm(A[:3, 3] - B[:3, 3]) < pos and angle_deg(A[:3, :3], B[:3, :3]) < rot_deg

    # ------------------------------------------------------------------ robot motion
    def blend_to(self, q_target, frames=None, joints=None):
        """joint-space move to q_target (only `joints` if given), smoothstep timing."""
        q0 = self.q.copy()
        idx = slice(None) if joints is None else self.s.jidx(joints)
        frames = max(frames or 0, int(np.ceil(np.max(np.abs(np.asarray(q_target)[idx] - q0[idx])) / MAX_STEP)), 6)
        for k in range(1, frames + 1):
            t = smooth(k / frames)
            q = q0.copy()
            q[idx] = (1 - t) * q0[idx] + t * np.asarray(q_target)[idx]
            self.q = q
            self.push()

    def reach(self, hand, target, frames=None, speed=0.15, others=None):
        """Cartesian path of one hand to target (world 4x4) by IK at every frame. others: {hand: (obj_pose_fn, rel)} hands
        that keep their grip on a shared object while it moves. Returns (ok, worst position error m)."""
        h = self.hands[hand]
        A = self.hand_pose(hand)
        dist = np.linalg.norm(target[:3, 3] - A[:3, 3]) + 0.1 * np.radians(angle_deg(A[:3, :3], target[:3, :3]))
        if frames is None:
            frames = int(np.clip(dist / speed * self.s.fps, 8, 60))
        worst = 0.0
        for k in range(1, frames + 1):
            P = interp_pose(A, target, smooth(k / frames))
            q_prev = self.q.copy()
            q, ep, er = self.kin.ik(h["link"], h["arm"], P, self.q, self.base())
            step = np.max(np.abs(q - q_prev))
            sub = int(np.ceil(step / MAX_STEP))
            for m in range(1, sub):                              # a big joint change (IK swinging a joint): spread it
                self.q = q_prev + (q - q_prev) * m / sub          # over extra frames (speed is not a constraint)
                self._follow(others)
                self.push()
            self.q = q
            ep = max(ep, self._follow(others))
            worst = max(worst, ep)
            self.push()
        return worst < 0.01, worst

    def solve(self, hand, target, seeds):
        """best IK solution for target over several seeds: (q, pos err, rot err)."""
        h = self.hands[hand]
        best = None
        for s in seeds:
            r = self.kin.ik(h["link"], h["arm"], target, s, self.base(), iters=300)
            if best is None or r[1] + 0.002 * r[2] < best[1] + 0.002 * best[2]:
                best = r
        return best

    def goto(self, hand, target, seeds=(), frames=None, others=None):
        """the hand to target: a straight hand path; if that gets stuck (joint limits), a joint-space move to the best IK
        solution from the given seeds (e.g. the demo's own pose for the same grip). Returns the final position error."""
        q0 = self.q.copy()
        n0 = self.n
        ok, err = self.reach(hand, target, frames=frames, others=others)
        if ok or not seeds:
            return err
        q, e, r = self.solve(hand, target, [q0, *seeds])
        if e >= err:
            return err
        for k in self.out:                                         # take the stuck path back out
            del self.out[k][n0:]
        for v in self.obj_out.values():
            del v[n0:]
        del self.bone_out[n0:]
        self.q = q0
        qt = q0.copy()
        idx = self.s.jidx(self.hands[hand]["arm"])
        qt[idx] = q[idx]
        self.blend_to(qt)
        return e

    def step_to(self, q):
        """set the joints to q; a jump bigger than MAX_STEP gets frames in between first."""
        q0 = self.q.copy()
        sub = int(np.ceil(np.max(np.abs(q - q0)) / MAX_STEP))
        for m in range(1, sub):
            self.q = q0 + (q - q0) * m / sub
            self.push()
        self.q = q

    def _follow(self, others):
        ep = 0.0
        for oh, (pose_fn, rel) in (others or {}).items():
            o = self.hands[oh]
            self.q, e2, _ = self.kin.ik(o["link"], o["arm"], pose_fn() @ np.linalg.inv(rel), self.q, self.base())
            ep = max(ep, e2)
        return ep

    def fingers(self, hand, q_profile, frames=8):
        idx = [i for i, n in enumerate(self.s.joints) if n.startswith(self.hands[hand]["fingers"])]
        q_profile = self.kin.clamp(q_profile)                     # robot limits: generated finger moves stay inside them
        q0 = self.q[idx].copy()
        for k in range(1, frames + 1):
            t = smooth(k / frames)
            self.q[idx] = (1 - t) * q0 + t * q_profile[idx]
            self.push()

    # ------------------------------------------------------------------ objects
    def attach(self, name, hands):
        P = self.obj_pose(name)
        self.free.pop(name, None)
        self.held[name] = {h: np.linalg.inv(self.hand_pose(h)) @ P for h in hands}
        self.take(name)

    def release(self, name, hand=None):
        if name not in self.held:
            return
        P = self.obj_pose(name)
        if hand is not None and len(self.held[name]) > 1:
            del self.held[name][hand]
            return
        del self.held[name]
        self.free[name] = P

    def detach_fall(self, name, landing, frames=None, keep_moving=None):
        """The object leaves the hand and falls to `landing` (pose resting on a surface). The robot keeps doing
        keep_moving(k) (e.g. the demo's next frames) while it falls."""
        P0 = self.obj_pose(name)
        self.held.pop(name, None)
        self.take(name)
        h = max(P0[2, 3] - landing[2, 3], 0.0)
        if frames is None:
            frames = max(int(np.ceil(np.sqrt(2 * h / G) * self.s.fps)) + 4, 6)
        tf = frames - 4
        for k in range(1, frames + 1):
            t = min(k / tf, 1.0)
            M = interp_pose(P0, landing, smooth(t))
            M[2, 3] = max(P0[2, 3] - 0.5 * G * (t * tf / self.s.fps) ** 2, landing[2, 3]) if t < 1 else landing[2, 3]
            self.free[name] = M
            if keep_moving:
                keep_moving(k)
            self.push()
        self.free[name] = landing
