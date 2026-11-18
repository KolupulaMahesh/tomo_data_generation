"""Keep the demo's own carry clear of the carton (user 2026-11-18: the carried phone >= 3 cm from it, the same phone
place kept). The phone's start place is random per seed, and for some places the demo's lift passes the open lid
closer than that. Here the demo itself is corrected before anything else uses it: around the close frames the holding
hand is lifted / moved away from the carton by a smooth bump (IK on that arm, the grip unchanged), just enough to keep
the gap (+ a small margin). Normal, single and multi episodes then all start from the corrected demo.
"""
import numpy as np

from .carry_gap import surface


def _carton_pts(Z, ks, a, c, fi):
    pts = []
    for k in ks:
        V = Z[f"vw_{k}"][fi].astype(float)
        T = Z[f"tri_{k}"]
        pts += [V, (V[T[:, 0]][:, None] * (1 - a - c)[None] + V[T[:, 1]][:, None] * a[None]
                    + V[T[:, 2]][:, None] * c[None]).reshape(-1, 3)]
    return np.vstack(pts)


def _gap(S, P, hs, ctr):
    """(distance m, closest carton point) from the object's box (pose P) to the carton points S."""
    L = (np.linalg.inv(P) @ np.c_[S, np.ones(len(S))].T).T[:, :3] - ctr
    d = np.linalg.norm(np.maximum(np.abs(L) - hs, 0), axis=1)
    j = int(d.argmin())
    return float(d[j]), S[j]


def fix(src, task, kin, log=print, margin=0.004):
    """correct src (in place) so every carried object with carton_gap_m keeps it from its grip until its place stage.
    Returns {grasp: {before_mm, after_mm, lift_mm, frames}} for the corrected ones."""
    scene = src.scene or "gidora_packing"
    holder = task["holder"]["object"]
    stages = dict((n, f) for n, f in task["stages"])
    try:
        efr, Z, ks, ta, tc = surface(scene, holder)
    except FileNotFoundError:
        return {}
    out = {}
    for key, g in task["grasps"].items():
        need = g.get("carton_gap_m")
        obj = g.get("object")
        if not need or not g.get("size_half") or obj not in src.objects or g.get("hand") not in ("right", "left"):
            continue
        hand = task["hands"][g["hand"]]
        hs, ctr = np.array(g["size_half"]), np.array(g.get("size_centre", [0, 0, 0]))
        g0, p0 = int(g["grip"]), int(stages.get(g.get("place_stage"), g["release"]))
        frames = [f for f in range(g0, p0) if f in efr]
        S = {f: _carton_pts(Z, ks, ta, tc, efr.index(f)) for f in frames}
        O = src.objects[obj]
        gaps = {f: _gap(S[f], O[f], hs, ctr) for f in frames}
        worst = min(frames, key=lambda f: gaps[f][0])
        if gaps[worst][0] >= need:
            continue
        # the window to correct: the frames closer than need + 3 cm, widened by a ramp each side
        close = [f for f in frames if gaps[f][0] < need + 0.03]
        ramp = 15
        a0, a1 = max(g0 + 3, close[0] - ramp), min(p0 - 1, close[-1] + ramp)
        # direction: up, and horizontally away from the carton point it comes closest to
        away = O[worst][:3, 3] - gaps[worst][1]
        away[2] = 0
        away = away / (np.linalg.norm(away) + 1e-9)
        dirn = away * 0.6 + np.array([0, 0, 0.8])
        dirn /= np.linalg.norm(dirn)
        rel = {f: np.linalg.inv(kin.link(hand["link"], src.q[f], src.robot_world[f])) @ O[f] for f in range(a0, a1 + 1)}
        q0, O0 = src.q.copy(), O.copy()
        best = None
        for amp in np.arange(0.01, 0.121, 0.01):
            q, Ob = q0.copy(), O0.copy()
            prev = q0[a0]
            for f in range(a0, a1 + 1):
                u = (f - a0) / max(a1 - a0, 1)
                w = 0.5 - 0.5 * np.cos(2 * np.pi * u)            # 0 -> 1 -> 0, smooth
                H = kin.link(hand["link"], q0[f], src.robot_world[f])
                H[:3, 3] += dirn * amp * w
                seed_q = q0[f].copy()
                idx = [src.joints.index(n) for n in hand["arm"]]
                seed_q[idx] = prev[idx]
                qf, ep, er = kin.ik(hand["link"], hand["arm"], H, seed_q, src.robot_world[f])
                q[f] = q0[f]
                q[f][idx] = qf[idx]
                prev = q[f]
                Ob[f] = kin.link(hand["link"], q[f], src.robot_world[f]) @ rel[f]
            g_after = min(_gap(S[f], Ob[f], hs, ctr)[0] for f in frames if a0 <= f <= a1)
            g_rest = min(gaps[f][0] for f in frames if not a0 <= f <= a1) if any(not a0 <= f <= a1 for f in frames) else 1
            best = (amp, q, Ob, min(g_after, g_rest))
            if best[3] >= need + margin:
                break
        amp, q, Ob, after = best
        src.q[:] = q
        O[:] = Ob
        for n, parent in (src.follows or {}).items():                # things hanging under the phone move with it
            if parent == obj and n in src.objects:
                for f in range(a0, a1 + 1):
                    src.objects[n][f] = Ob[f] @ np.linalg.inv(O0[f]) @ src.objects[n][f]
        out[key] = {"before_mm": round(gaps[worst][0] * 1000, 1), "after_mm": round(after * 1000, 1),
                    "lift_mm": round(amp * 1000), "frames": [a0, a1]}
        log(f"[carry_fix] {key}: the demo's carry came {out[key]['before_mm']} mm from the carton (frame {worst}) -> "
            f"hand moved up / away up to {out[key]['lift_mm']} mm over frames {a0}-{a1}: now {out[key]['after_mm']} mm")
    return out
