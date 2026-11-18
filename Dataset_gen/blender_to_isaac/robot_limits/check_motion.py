"""Check a motion against what the REAL robot allows (tomo_re_uhand_limits.yaml + the robot's collision meshes):
joint positions, speed, acceleration and self-collision (any two links touching that the SRDF does not allow, e.g.
two fingers of one hand). Read-only.

    ../env/bin/python check_motion.py <motion> [--fps 30] [--stride 1] [--segments segments.json] [--out report.json]

--scene <name> checks against that scene's robot (scene_robot.py: its URDF + SRDF, limits = the URDF's with the same 2 deg margin; the speed
caps stay the first robot's brain caps until the robot's own are known). Default: the first robot.
<motion> = an .npz with names + q (e.g. ../renderer/isaac_base/base_motion.npz), an episode folder
(q.npy + joint_names.json [+ segments.json]), a dataset folder of episode folders, or an MMRT recording (.mmrt.yaml).
Exit code 0 = the robot can do it, 1 = it breaks a hard rule (outside a limit, faster than the cap, or a collision).
"""
import argparse
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE / "libs"))                       # trimesh + python-fcl (pip --target), after the venv's numpy
import fcl                                                 # noqa: E402
import trimesh                                             # noqa: E402

SRC = HERE / "source"
LIMITS = yaml.safe_load((HERE / "tomo_re_uhand_limits.yaml").read_text())
sys.path.insert(0, str(HERE.parent))
from scene_robot import FIRST_ROBOT_URDF, scene_setup, urdf_limits  # noqa: E402


# ---------------- loading a motion ----------------
def load_motion(path):
    """-> list of (name, joint names, q [T, J], segments or None)."""
    p = Path(path)
    if p.suffix == ".npz":
        z = np.load(p)
        return [(p.name, [str(n) for n in z["names"]], z["q"].astype(np.float64), None)]
    if p.name.endswith(".mmrt.yaml"):
        d = yaml.safe_load(p.read_text())
        return [(p.name, [x for v in d["labels"].values() for x in v], np.asarray(d["frames"], np.float64), None)]
    if (p / "q.npy").exists():
        return [episode(p)]
    eps = sorted(e for e in p.iterdir() if (e / "q.npy").exists())
    if not eps:
        raise SystemExit(f"no motion found in {p}")
    return [episode(e) for e in eps]


def episode(e):
    names = json.loads((e / "joint_names.json").read_text())
    names = names if isinstance(names, list) else names.get("names", names)
    seg = json.loads((e / "segments.json").read_text())["segments"] if (e / "segments.json").exists() else None
    return e.name, names, np.load(e / "q.npy").astype(np.float64), seg


# ---------------- robot model: kinematics + collision meshes ----------------
def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def origin(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, :3] = rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return T


def axis_rot(ax, a):
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


class Robot:
    def __init__(self, urdf=SRC / "robot_description.urdf", srdf=SRC / "tomo_config.srdf"):
        self.base = Path(urdf).parent
        u = ET.parse(urdf).getroot()
        self.joints = []
        for j in u.iter("joint"):
            ax = np.array([float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0").split()])
            mim = j.find("mimic")
            self.joints.append(dict(name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
                                    child=j.find("child").get("link"), T=origin(j), axis=ax / (np.linalg.norm(ax) or 1.0),
                                    mimic=(mim.get("joint"), float(mim.get("multiplier", 1)), float(mim.get("offset", 0)))
                                    if mim is not None else None))
        children = {j["child"] for j in self.joints}
        self.root = next(l.get("name") for l in u.iter("link") if l.get("name") not in children)
        order, todo = [], [self.root]                       # joints parent-first
        while todo:
            link = todo.pop()
            for j in self.joints:
                if j["parent"] == link:
                    order.append(j); todo.append(j["child"])
        self.joints = order
        # collision objects: one per <collision> element
        self.objs = []
        for l in u.iter("link"):
            for c in l.findall("collision"):
                mesh = self._mesh(c.find("geometry"))
                if mesh is None:
                    continue
                bvh = fcl.BVHModel(); bvh.beginModel(len(mesh.vertices), len(mesh.faces))
                bvh.addSubModel(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32)); bvh.endModel()
                lo, hi = mesh.bounds
                corners = np.array(list(itertools.product(*zip(lo, hi))))
                self.objs.append(dict(link=l.get("name"), obj=fcl.CollisionObject(bvh), T=origin(c), corners=corners))
        off = {frozenset((d.get("link1"), d.get("link2"))) for d in ET.parse(srdf).getroot().iter("disable_collisions")}
        pairs = [(a, b) for a, b in itertools.combinations(range(len(self.objs)), 2)
                 if self.objs[a]["link"] != self.objs[b]["link"]
                 and frozenset((self.objs[a]["link"], self.objs[b]["link"])) not in off]
        self.pa, self.pb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        # gaps with every joint at 0: neighbouring fingers sit ~5 mm apart by design, so a pair only counts as too close
        # when the motion brings it nearer than it is at rest
        self.rest_gap = {}
        self.rest_gap = self.collisions({}, 0.05)[1]

    def _mesh(self, g):
        g = g[0]
        if g.tag == "mesh":
            f = g.get("filename")
            f = self.base / (f[len("package://"):] if f.startswith("package://") else f)
            m = trimesh.load(f, force="mesh")
            if g.get("scale"):
                m.apply_scale([float(v) for v in g.get("scale").split()])
            return m
        if g.tag == "box":
            return trimesh.creation.box(extents=[float(v) for v in g.get("size").split()])
        if g.tag == "cylinder":
            return trimesh.creation.cylinder(radius=float(g.get("radius")), height=float(g.get("length")))
        if g.tag == "sphere":
            return trimesh.creation.icosphere(radius=float(g.get("radius")))
        return None

    def link_poses(self, qd):
        W = {self.root: np.eye(4)}
        for j in self.joints:
            q = qd.get(j["name"], 0.0)
            if j["mimic"]:
                q = j["mimic"][1] * qd.get(j["mimic"][0], 0.0) + j["mimic"][2]
            M = np.eye(4)
            if j["type"] in ("revolute", "continuous"):
                M[:3, :3] = axis_rot(j["axis"], q)
            elif j["type"] == "prismatic":
                M[:3, 3] = j["axis"] * q
            W[j["child"]] = W[j["parent"]] @ j["T"] @ M
        return W

    def collisions(self, qd, clearance=0.0):
        """-> (set of (linkA, linkB) touching, {(linkA, linkB): gap m} for pairs closer than clearance) in this pose."""
        W = self.link_poses(qd)
        lo, hi = np.empty((len(self.objs), 3)), np.empty((len(self.objs), 3))
        for i, o in enumerate(self.objs):
            T = W[o["link"]] @ o["T"]
            o["obj"].setTransform(fcl.Transform(T[:3, :3], T[:3, 3]))
            c = o["corners"] @ T[:3, :3].T + T[:3, 3]
            lo[i], hi[i] = c.min(0), c.max(0)
        near = np.flatnonzero(np.all(lo[self.pa] <= hi[self.pb] + clearance, 1) & np.all(lo[self.pb] <= hi[self.pa] + clearance, 1))
        hits, close = set(), {}
        for k in near:
            a, b = self.objs[self.pa[k]], self.objs[self.pb[k]]
            pair = tuple(sorted((a["link"], b["link"])))
            if fcl.collide(a["obj"], b["obj"], fcl.CollisionRequest(), fcl.CollisionResult()):
                hits.add(pair)
            elif clearance > 0:
                d = fcl.distance(a["obj"], b["obj"], fcl.DistanceRequest(), fcl.DistanceResult())
                if d < min(clearance, self.rest_gap.get(pair, np.inf) - 0.001):
                    close[pair] = min(d, close.get(pair, d))
        return hits, close


# ---------------- checks ----------------
def spans(frames):
    """[3,4,5,9] -> '3-5, 9'"""
    out, s = [], None
    for i, f in enumerate(frames):
        if s is None:
            s = f
        if i + 1 == len(frames) or frames[i + 1] != f + 1:
            out.append(f"{s}-{f}" if f > s else f"{s}"); s = None
    return ", ".join(out[:8]) + (f" (+{len(out) - 8} more)" if len(out) > 8 else "")


def step_of(seg, f):
    return next((s["name"] for s in seg or [] if s["start"] <= f <= s["end"]), "")


def check(name, names, q, seg, robot, fps, stride, clearance_mm, J):
    known = [i for i, n in enumerate(names) if n in J]
    rep = {"motion": name, "frames": len(q), "position": {}, "speed": {}, "acceleration": {}, "collision": {}}
    for i in known:
        n, lim = names[i], J[names[i]]
        out = np.flatnonzero((q[:, i] < lim["lower_rad"] - 1e-6) | (q[:, i] > lim["upper_rad"] + 1e-6))
        near = np.flatnonzero((q[:, i] < lim["safe_lower_rad"]) | (q[:, i] > lim["safe_upper_rad"]))
        if len(out) or len(near):
            rep["position"][n] = dict(outside=len(out), outside_frames=spans(out.tolist()), within_2deg=len(near),
                                      min_deg=round(math.degrees(q[:, i].min()), 2), max_deg=round(math.degrees(q[:, i].max()), 2),
                                      limits_deg=[lim["lower_deg"], lim["upper_deg"]])
    cap, normal, acap = (LIMITS["brain_stream"][k] for k in ("stream_max_speed_rad_s", "stream_speed_rad_s", "stream_max_accel_rad_s2"))
    V = np.abs(np.diff(q, axis=0)) * fps
    A = np.abs(np.diff(q, 2, axis=0)) * fps * fps
    for i in known:
        fast = np.flatnonzero(V[:, i] > cap)
        if len(fast):
            k = int(V[:, i].argmax())
            rep["speed"][names[i]] = dict(over_cap=len(fast), over_normal=int((V[:, i] > normal).sum()),
                                          max_deg_s=round(math.degrees(V[k, i])), at_frame=k, step=step_of(seg, k))
        hard = int((A[:, i] > acap).sum())
        if hard:
            rep["acceleration"][names[i]] = dict(over=hard, max_deg_s2=round(math.degrees(A[:, i].max())))
    hits, close = {}, {}
    for f in range(0, len(q), stride):
        h, c = robot.collisions(dict(zip(names, q[f])), clearance_mm / 1000)
        for pair in h:
            hits.setdefault(pair, []).append(f)
        for pair, d in c.items():
            close.setdefault(pair, []).append((f, d))
    for pair, fr in sorted(hits.items(), key=lambda kv: kv[1][0]):
        rep["collision"][" vs ".join(pair)] = dict(frames=len(fr), where=spans(fr), step=step_of(seg, fr[0]))
    rep["too_close"] = {}
    for pair, fd in sorted(close.items(), key=lambda kv: kv[1][0][0]):
        f_min, d_min = min(fd, key=lambda x: x[1])
        rep["too_close"][" vs ".join(pair)] = dict(frames=len(fd), where=spans([f for f, _ in fd]),
                                                   min_gap_mm=round(1000 * d_min, 1), at_frame=f_min, step=step_of(seg, f_min))
    rep["robot_can_do_it"] = not (any(v["outside"] for v in rep["position"].values()) or rep["speed"] or rep["collision"])
    return rep


def show(rep):
    print(f"\n=== {rep['motion']}: {rep['frames']} frames -> {'OK' if rep['robot_can_do_it'] else 'NOT OK for the robot'}")
    out = {n: v for n, v in rep["position"].items() if v["outside"]}
    print(f"  POSITION: {len(out)} joints outside the robot's limits" + ("" if out else " (ok)"))
    for n, v in out.items():
        print(f"    {n:18s} {v['outside']:4d} frames ({v['outside_frames']}), went {v['min_deg']}..{v['max_deg']} deg, limits {v['limits_deg']}")
    close = [n for n, v in rep["position"].items() if not v["outside"]]
    if close:
        print(f"    (inside the limits but closer than 2 deg: {', '.join(close)})")
    print(f"  SPEED: {len(rep['speed'])} joints faster than the robot's cap" + ("" if rep["speed"] else " (ok)"))
    for n, v in sorted(rep["speed"].items(), key=lambda kv: -kv[1]["max_deg_s"])[:12]:
        print(f"    {n:18s} {v['over_cap']:4d} frames over the cap, fastest {v['max_deg_s']} deg/s at frame {v['at_frame']} {v['step']}")
    if len(rep["speed"]) > 12:
        print(f"    ... and {len(rep['speed']) - 12} more joints")
    print(f"  ACCELERATION (soft, the robot smooths it): {len(rep['acceleration'])} joints over the robot's 57 deg/s^2")
    print(f"  SELF-COLLISION: {len(rep['collision'])} touching pairs" + ("" if rep["collision"] else " (ok)"))
    for pair, v in rep["collision"].items():
        print(f"    {pair:40s} {v['frames']:4d} frames ({v['where']}) {v['step']}")
    if "too_close" in rep:
        groups = {"robot vs itself": {k: v for k, v in rep["too_close"].items() if "workcell_table" not in k},
                  "robot vs table": {k: v for k, v in rep["too_close"].items() if "workcell_table" in k}}
        for g, items in groups.items():
            print(f"  TOO CLOSE, {g} (warning: a small model error makes them touch): {len(items)} pairs")
            for pair, v in sorted(items.items(), key=lambda kv: kv[1]["min_gap_mm"])[:8]:
                print(f"    {pair:40s} {v['frames']:4d} frames ({v['where']}), closest {v['min_gap_mm']} mm at frame {v['at_frame']} {v['step']}")
            if len(items) > 8:
                print(f"    ... and {len(items) - 8} more pairs (all in the --out report)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("motion")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--stride", type=int, help="collision check every Nth frame (default: the scene's limits_stride, else 1)")
    ap.add_argument("--clearance-mm", type=float, default=10.0,
                    help="warn when two links that may not touch come nearer than this (0 = off)")
    ap.add_argument("--segments", help="segments.json for step names (an episode folder brings its own)")
    ap.add_argument("--out", help="write the full report as JSON")
    ap.add_argument("--scene", help="check against this scene's robot (default: the first robot)")
    a = ap.parse_args()
    setup = scene_setup(a.scene) if a.scene else None
    robot = Robot(setup["urdf"], setup["srdf"]) if setup else Robot()
    stride = a.stride or (setup["limits_stride"] if setup else 1)
    joints = urdf_limits(setup["urdf"], LIMITS["safe_margin_deg"]) if setup and setup["urdf"] != str(FIRST_ROBOT_URDF) else LIMITS["joints"]
    seg = json.loads(Path(a.segments).read_text())["segments"] if a.segments else None
    reps = []
    for name, names, q, s in load_motion(a.motion):
        rep = check(name, names, q, s or seg, robot, a.fps, stride, a.clearance_mm, joints)
        show(rep); reps.append(rep)
    if len(reps) > 1:
        print(f"\n{sum(r['robot_can_do_it'] for r in reps)}/{len(reps)} motions OK for the robot")
    if a.out:
        Path(a.out).write_text(json.dumps(reps if len(reps) > 1 else reps[0], indent=1))
    sys.exit(0 if all(r["robot_can_do_it"] for r in reps) else 1)


if __name__ == "__main__":
    main()
