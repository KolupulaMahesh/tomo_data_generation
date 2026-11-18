"""The demo an episode is built from: per frame the robot's joints, object world poses, carton / tape bones and the IO byte.

Two sources, same interface:
  - MMRT recording (boxing_001.mmrt.yaml): robot joints + the phone. Enough for the phone scenarios and for testing.
  - blender_to_isaac motion.npz (make_motion.py): every moving object and bone of the scene. Needed for the carton / tape
    scenarios and for rendering in Isaac Sim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

B2I = Path(__file__).resolve().parents[2] / "Dataset_gen/blender_to_isaac"


def quat_to_mat(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass
class Source:
    joints: list[str]                       # robot joint order
    q: np.ndarray                           # (T, J) radians
    robot_world: np.ndarray                 # (T, 4, 4) robot base in the world
    objects: dict[str, np.ndarray]          # name -> (T, 4, 4) world pose
    relay: np.ndarray                       # (T,) IO byte, io0..io7 = bits 0..7
    fps: float = 30.0
    bones: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)   # (armature, bone) -> (T, 4, 4) armature space
    kind: str = "mmrt"
    path: str = ""
    robot_attached: set = field(default_factory=set)        # objects riding on the robot (its link meshes, camera):
                                                            # their pose comes from the joints (export.py), not tracked here
    follows: dict = field(default_factory=dict)             # child -> top object it hangs under (moves with it)
    bone_parent: dict = field(default_factory=dict)         # (armature, bone) -> its parent bone (None: root)
    scene: str = ""                                         # blender_to_isaac scene name (its blender_export folder)

    @property
    def T(self):
        return len(self.q)

    def jidx(self, names):
        return [self.joints.index(n) for n in names]


def load_mmrt(path, bit_order="msb_first") -> Source:
    d = yaml.load(Path(path).read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    rob, act = d["labels"]["robots"], d["labels"]["active_objects"]
    F = np.asarray(d["frames"], float)
    root = rob[0].split(".")[0]
    joints = [n for n in rob if not n.startswith(root + ".")]
    nr = len(rob)
    q = F[:, 7:nr]
    T = len(F)
    RW = np.tile(np.eye(4), (T, 1, 1))
    RW[:, :3, 3] = F[:, 0:3]
    for i in range(T):
        RW[i, :3, :3] = quat_to_mat(*F[i, 3:7])
    objects = {}
    names = sorted({a.split(".location")[0].split(".rotation")[0] for a in act}, key=lambda n: act.index(n + ".location.x"))
    for k, n in enumerate(names):
        c = nr + 7 * k
        W = np.tile(np.eye(4), (T, 1, 1))
        W[:, :3, 3] = F[:, c:c + 3]
        for i in range(T):
            W[i, :3, :3] = quat_to_mat(*F[i, c + 3:c + 7])
        objects[n] = W
    relay = np.zeros(T, np.uint8)
    for f, v, _ in sorted(d.get("io_events") or []):
        v = int(v)
        if bit_order == "msb_first":                      # MMRT writes io0 as the top bit
            v = int(f"{v:08b}"[::-1], 2)
        relay[int(f):] = v
    return Source(joints, q, RW, objects, relay, float(d["metadata"].get("frame_rate", 30.0)), kind="mmrt", path=str(path))


def load_motion(path) -> Source:
    M = np.load(path)
    joints = [str(x) for x in M["robot_joints"]]
    objects = {str(n): M["object_world"][:, k] for k, n in enumerate(M["object_names"])}
    bones = {(str(a), str(b)): M["bone_pose"][:, k] for k, (a, b) in enumerate(zip(M["bone_owner"], M["bone_names"]))}
    fps = float(M["fps"]) if "fps" in M.files else 30.0
    s = Source(joints, M["robot_q"].astype(float), M["robot_world"].astype(float), objects, M["relay"].astype(np.uint8),
               fps, bones, kind="motion", path=str(path))
    ep = json.loads((Path(path).parent / "episode.json").read_text())
    s.scene = ep["scene"]
    rig = json.loads((B2I / "blender_export" / ep["scene"] / "rig.json").read_text())["objects"]
    s.bone_parent = {(a, b): rig[a]["bones"][b]["parent"] for (a, b) in bones if a in rig and b in rig[a].get("bones", {})}
    for n in objects:
        chain, p = [], rig.get(n, {}).get("parent")
        while p:
            chain.append(p)
            p = rig.get(p, {}).get("parent")
        if ep["robot"] in chain:
            s.robot_attached.add(n)
        elif chain and chain[-1] in objects:
            s.follows[n] = chain[-1]
    return s


def smooth_elbow(src, kin, task, max_deg=16.0, pad=12, log=print, frames=(100, 200)):
    """Blender's IK can swing an arm's elbow round (the 7th joint lets the elbow move without moving the hand) by more
    than the demo ever turns a joint in one frame (14.8 deg). Where that happens, the arm is solved again for the SAME
    hand poses, with the elbow's change spread over the frames around it: the hand (and what it holds) does not move,
    only the elbow takes the smooth way. Returns the number of frames changed."""
    q = src.q
    thr = np.radians(max_deg)
    changed = 0
    for h, hc in task["hands"].items():
        idx = [src.joints.index(n) for n in hc["arm"]]
        step = np.r_[0.0, np.abs(np.diff(q[:, idx], axis=0)).max(1)]
        bad = np.flatnonzero(step > thr)
        bad = bad[(bad >= frames[0]) & (bad < frames[1])]           # only where the randomisation changed the motion
        if not len(bad):
            continue
        groups = np.split(bad, np.flatnonzero(np.diff(bad) > pad) + 1)
        for g in groups:
            A, B = max(int(g[0]) - pad, 0), min(int(g[-1]) + pad, src.T - 1)
            qa, qb = q[A, idx].copy(), q[B, idx].copy()
            new = {}
            for f in range(A + 1, B):
                t = (f - A) / (B - A)
                t = t * t * (3 - 2 * t)
                seed = q[f].copy()
                seed[idx] = (1 - t) * qa + t * qb
                tgt = kin.link(hc["link"], q[f], src.robot_world[f])
                qn, ep, er = kin.ik(hc["link"], hc["arm"], tgt, seed, src.robot_world[f], iters=300)
                if ep > 0.003 or er > 2.0:                       # 3 mm: invisible, the phone in the hand moves with it
                    log(f"[source] {h} arm: elbow swing at {int(g[0])}-{int(g[-1])} kept (IK {ep * 1000:.1f} mm / {er:.1f} deg off at {f})")
                    break
                new[f] = qn
            else:
                for f, qn in new.items():
                    q[f] = qn
                changed += len(new)
                log(f"[source] {h} arm: elbow swing at frames {int(g[0])}-{int(g[-1])} "
                    f"({np.degrees(step[g].max()):.0f} deg/frame) spread over {A}-{B}, hand unchanged")
    return changed


def load(task, motion=None) -> Source:
    s = task["source"]
    if motion or s.get("motion"):
        return load_motion(motion or s["motion"])
    return load_mmrt(s["mmrt"], s.get("mmrt_io_bit_order", "msb_first"))
