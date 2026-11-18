"""Final verification of an episode from its frames (the scene's actual state, not what was planned).

An episode is successful only if every check passes: each object ends where the demo leaves it, nothing is still held,
every carton part ends folded, the robot never leaves its joint limits, and no object ever jumps.
"""
from __future__ import annotations

import numpy as np

from . import detect as D
from .kin import angle_deg


def run(b, task, tol, grasps, meta):
    checks = {}
    T = b.n
    last = b.out["src"][-1]
    complete = last >= b.s.T - 1 and not meta["terminal"]
    checks["task_reached_end"] = bool(complete)

    # each object vs where the demo leaves it at the end
    for name, W in b.s.objects.items():
        if name in b.s.robot_attached or name in b.s.follows:       # robot parts: from the joints; children: with parent
            continue
        P = b.obj_out[name][-1]
        e, r = D.placed_error(P, W[-1])
        checks[f"{name}_final_pose"] = {"ok": bool(e < tol["placed_pos_m"] and r < tol["placed_rot_deg"]),
                                        "pos_err_m": round(e, 4), "rot_err_deg": round(r, 2)}

    # the stage results the task needs (object vs its target right after the stage)
    for k, g in grasps.items():
        if not g.available:
            continue
        f = min(g.release + g.settle, b.s.T - 1)                    # the stage's result, once the task runs normally again
        src, ph = np.array(b.out["src"]), np.array(b.out["phase"])
        idx = np.flatnonzero((src >= f) & (ph == "normal"))
        if len(idx):
            P = b.obj_out[g.obj][idx[0]]
            g_t = b.s.objects[g.obj][src[idx[0]]]
            e, r = D.placed_error(P, g_t)
            checks[f"{k}_placed_after_stage"] = {"ok": bool(e < tol["placed_pos_m"] and r < tol["placed_rot_deg"]),
                                                 "pos_err_m": round(e, 4), "rot_err_deg": round(r, 2)}
    for k, p in (task.get("carton_parts") or {}).items():
        if p["bones"] and all((p["armature"], x) in b.s.bones for x in p["bones"]):
            i = T - 1
            worst = 0.0
            for x in p["bones"]:
                key = (p["armature"], x)
                cur = b.bone_out[i].get(key, b.s.bones[key][b.out["src"][i]])
                worst = max(worst, angle_deg(cur[:3, :3], b.s.bones[key][-1][:3, :3]))
            checks[f"{k}_folded"] = {"ok": worst < tol["flap_deg"], "err_deg": round(worst, 1)}

    checks["nothing_held_at_end"] = not b.held
    # ROBOT LIMITS: every frame inside the hard limits; every generated frame inside them too (margin 0), except where it is
    # exactly as close as the demo itself is at that frame (blending back onto the demo)
    qa, sa = np.array(b.out["q"]), np.array(b.out["src"])
    ok, bad = b.kin.within_limits(qa)
    checks["joint_limits"] = {"ok": ok, "outside": bad[:10]}
    dq = b.s.q[sa]
    gen_rows = ~np.all(np.isclose(qa, dq), axis=1)
    margin_bad = []
    for i, n in enumerate(b.kin.joints):
        if n not in b.kin.lim_safe or not gen_rows.any():
            continue
        lo, hi = b.kin.lim_safe[n]
        v, d = qa[gen_rows, i], dq[gen_rows, i]
        over = (v < np.minimum(lo, d) - 1e-6) | (v > np.maximum(hi, d) + 1e-6)
        if over.any():
            margin_bad.append([n, int(over.sum()), round(float(np.degrees(min(v.min() - b.kin.lim[n][0], b.kin.lim[n][1] - v.max()))), 2)])
    checks["joint_limit_margin"] = {"ok": not margin_bad, "margin_deg": round(float(np.degrees(b.kin.margin)), 1),
                                    "too_close": margin_bad[:10]}

    # no jumps: an object not carried by a hand moves at most max_step_m per frame
    jumps = []
    for name, L in b.obj_out.items():
        if name in b.s.robot_attached:
            continue
        P = np.array([x[:3, 3] for x in L])
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)
        for i in np.flatnonzero(step > tol["max_step_m"]):
            jumps.append([name, int(i + 1), round(float(step[i]), 3)])
    checks["no_object_jumps"] = {"ok": not jumps, "jumps": jumps[:10]}
    # joint jumps in the frames the generator made (the demo's own frames are as the demo is: its left thumb moves
    # 0.6 rad in one frame at 192)
    qs = np.array(b.out["q"])
    src = np.array(b.out["src"])
    is_demo = np.all(np.isclose(qs, b.s.q[src]), axis=1)          # this frame's joints are exactly the demo's
    demo_step = (np.diff(src) == 1) & is_demo[1:] & is_demo[:-1]
    dq = np.abs(np.diff(qs, axis=0)).max(1) if T > 1 else np.zeros(0)
    gen = dq[~demo_step] if (~demo_step).any() else np.zeros(1)
    # the arms never touch in the frames the generator made (the demo itself keeps them >= 24 mm apart)
    gen_idx = np.flatnonzero(~is_demo)[::2]
    gaps = [(b.clear.gap(qs[i], b.s.robot_world[src[i]]), int(i)) for i in gen_idx] if hasattr(b, "clear") else []
    worst = min(gaps) if gaps else (1.0, -1)
    checks["arm_clearance"] = {"ok": bool(worst[0] >= tol.get("min_arm_gap_generated_m", 0.015)),
                               "min_gap_mm": round(worst[0] * 1000, 1), "at_frame": worst[1]}
    checks["no_joint_jumps"] = {"ok": bool(gen.max() < 0.35), "max_step_rad_generated": round(float(gen.max()), 3)}

    # a carried object keeps its distance from the carton (user 2026-11-18: 3 cm) in the frames the generator made, from its
    # grip until its placing starts (then it goes into the carton on the demo's own path)
    for k, g in grasps.items():
        need = g.c.get("carton_gap_m") if g.available else None
        if not need or not g.c.get("size_half"):
            continue
        place0 = dict((n, f) for n, f in task["stages"]).get(g.c.get("place_stage"), g.release)
        idx = [i for i in range(len(src)) if g.grip <= src[i] < place0]    # every carry frame, the demo's too (user 2026-11-18)
        if not idx:
            continue
        gap, at = carton_gap(b, task, g, idx)
        checks[f"{k}_carton_gap"] = {"ok": bool(gap >= need), "min_gap_mm": round(gap * 1000, 1), "at_frame": at}

    def passed(v):
        return v if isinstance(v, bool) else v["ok"]

    success = complete and all(passed(v) for v in checks.values())
    return {"success": bool(success), "failed_checks": [k for k, v in checks.items() if not passed(v)], "checks": checks}


_SURF = {}


def carton_gap(b, task, g, frames, per_tri=8):
    """smallest distance (m) from the object's box to the carton's surface over these output frames, and where. The
    carton surface: its mesh as the demo bends it at that demo frame (objects export), moved to where the carton is."""
    import json
    from .export import B2I
    arm = task["holder"]["object"]
    scene = getattr(b.s, "scene", "") or "gidora_packing"
    if scene not in _SURF:
        Z = np.load(B2I / "blender_export" / scene / "objects_export.npz")
        info = json.loads(str(Z["info"]))
        ks = [i for i, o in enumerate(info["objects"]) if arm in o.get("armature", []) and f"vw_{i}" in Z.files]
        rng = np.random.default_rng(0)
        a, c = rng.random((2, per_tri, 1))
        m = a + c > 1
        a[m], c[m] = 1 - a[m], 1 - c[m]
        _SURF[scene] = (Z, ks, a, c)
    Z, ks, a, c = _SURF[scene]
    hs = np.array(g.c["size_half"])
    ctr = np.array(g.c.get("size_centre", [0.0, 0.0, 0.0]))
    best = (1.0, -1)
    for i in frames[::2]:
        s = int(b.out["src"][i])
        fi = min(s, Z[f"vw_{ks[0]}"].shape[0] - 1) if ks else 0
        pts = []
        for k in ks:
            V = Z[f"vw_{k}"][fi].astype(float)
            T = Z[f"tri_{k}"]
            pts.append((V[T[:, 0]][:, None] * (1 - a - c)[None] + V[T[:, 1]][:, None] * a[None]
                        + V[T[:, 2]][:, None] * c[None]).reshape(-1, 3))
            pts.append(V)
        if not pts:
            continue
        S = np.vstack(pts)
        Wd, Wn = b.s.objects[arm][s], b.obj_out[arm][i]                 # demo -> where the carton is now
        S = (Wn @ np.linalg.inv(Wd) @ np.c_[S, np.ones(len(S))].T).T
        L = (np.linalg.inv(b.obj_out[g.obj][i]) @ S.T).T[:, :3] - ctr
        d = float(np.linalg.norm(np.maximum(np.abs(L) - hs, 0), axis=1).min())
        if d < best[0]:
            best = (d, int(i))
    return best
