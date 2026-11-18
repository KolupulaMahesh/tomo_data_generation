"""Editing F-curves the way Blender 5.2 does when a key is inserted (animrig insert_vert_fcurve / insert_bezt_fcurve)
and its handles recalculated (BKE_fcurve_handles_recalc_ex: calchandleNurb_intern, then for 'Continuous Acceleration'
curves BKE_nurb_handle_smooth_fcurve). Works on the key dicts of rig.json (fcurves.FCurve.keys) in place.
Pure numpy / Python, so it runs in Isaac Sim's Python.
"""
from __future__ import annotations

import math

import numpy as np

AUTO = ("AUTO", "AUTO_CLAMPED")
BINARYSEARCH_THRESH = 0.01                       # BEZT_BINARYSEARCH_THRESH
FLT_MAX = 3.4028234663852886e38


def _f32(x):
    return float(np.float32(x))


# ------------------------------------------------------------------ key insertion
def find_key(keys, frame):
    """BKE_fcurve_bezt_binarysearch_index: (index, replace) for a key at `frame`."""
    for i, k in enumerate(keys):
        if abs(k["co"][0] - frame) < BINARYSEARCH_THRESH:
            return i, True
        if k["co"][0] > frame:
            return i, False
    return len(keys), False


def insert_key(keys, frame, value, interpolation="BEZIER", handle_type="AUTO_CLAMPED"):
    """insert_vert_fcurve without INSERTKEY_REPLACE / FAST: a key already at the frame keeps its interpolation and
    handle types and only moves by the value change (its handles with it); a new key gets the user-preference types,
    and the interpolation of the key before it (after it, if first) once the curve has more than two keys."""
    value = _f32(value)
    i, replace = find_key(keys, frame)
    if replace:
        k = keys[i]
        dy = value - k["co"][1]
        k["co"][1] = value
        k["handle_left"][1] += dy
        k["handle_right"][1] += dy
        return i
    keys.insert(i, {"co": [float(frame), value], "interpolation": interpolation, "easing": "AUTO",
                    "handle_left": [frame - 1.0, value], "handle_right": [frame + 1.0, value],
                    "handle_left_type": handle_type, "handle_right_type": handle_type,
                    "back": 1.70158, "amplitude": 0.8, "period": 4.1, "type": "KEYFRAME"})
    if len(keys) > 2:
        if i > 0:
            keys[i]["interpolation"] = keys[i - 1]["interpolation"]
        elif i < len(keys) - 1:
            keys[i]["interpolation"] = keys[i + 1]["interpolation"]
    return i


# ------------------------------------------------------------------ handles: first pass (curve.cc)
def _calc_handle(b, prev, nxt, smoothing):
    """calchandleNurb_intern for an F-curve key (is_fcurve, not skip_align)."""
    b["_auto"] = "NORMAL"
    h1, h2 = b["handle_left_type"], b["handle_right_type"]
    if h1 == "FREE" and h2 == "FREE":
        return
    p2 = np.array(b["co"], float)
    hl, hr = np.array(b["handle_left"], float), np.array(b["handle_right"], float)
    p1 = np.array(prev["co"], float) if prev is not None else None
    p3 = np.array(nxt["co"], float) if nxt is not None else None
    if p1 is None:
        p1 = 2.0 * p2 - p3
    if p3 is None:
        p3 = 2.0 * p2 - p1
    dvec_a, dvec_b = p2 - p1, p3 - p2
    len_a, len_b = dvec_a[0], dvec_b[0]
    if len_a == 0.0:
        len_a = 1.0
    if len_b == 0.0:
        len_b = 1.0
    if h1 in AUTO or h2 in AUTO:
        tvec = dvec_b / len_b + dvec_a / len_a
        ln = (6.0 / 2.5614 if smoothing != "NONE" else tvec[0]) * 2.5614
        if ln != 0.0:
            leftviolate = rightviolate = False
            if smoothing == "NONE":
                if len_a > 5.0 * len_b:
                    len_a = 5.0 * len_b
                if len_b > 5.0 * len_a:
                    len_b = 5.0 * len_a
            if h1 in AUTO:
                len_a /= ln
                hl = p2 - tvec * len_a
                if h1 == "AUTO_CLAMPED" and nxt is not None and prev is not None:
                    yd1, yd2 = prev["co"][1] - p2[1], nxt["co"][1] - p2[1]
                    if (yd1 <= 0 and yd2 <= 0) or (yd1 >= 0 and yd2 >= 0):
                        hl[1] = p2[1]
                        b["_auto"] = "LOCKED"
                    elif yd1 <= 0:
                        if prev["co"][1] > hl[1]:
                            hl[1], leftviolate = prev["co"][1], True
                    elif prev["co"][1] < hl[1]:
                        hl[1], leftviolate = prev["co"][1], True
            if h2 in AUTO:
                len_b /= ln
                hr = p2 + tvec * len_b
                if h2 == "AUTO_CLAMPED" and nxt is not None and prev is not None:
                    yd1, yd2 = prev["co"][1] - p2[1], nxt["co"][1] - p2[1]
                    if (yd1 <= 0 and yd2 <= 0) or (yd1 >= 0 and yd2 >= 0):
                        hr[1] = p2[1]
                        b["_auto"] = "LOCKED"
                    elif yd1 <= 0:
                        if nxt["co"][1] < hr[1]:
                            hr[1], rightviolate = nxt["co"][1], True
                    elif nxt["co"][1] > hr[1]:
                        hr[1], rightviolate = nxt["co"][1], True
            if leftviolate or rightviolate:
                h1_x, h2_x = hl[0] - p2[0], p2[0] - hr[0]
                if leftviolate:
                    hr[1] = p2[1] + ((p2[1] - hl[1]) / h1_x) * h2_x
                else:
                    hl[1] = p2[1] + ((p2[1] - hr[1]) / h2_x) * h1_x
    if h1 == "VECTOR":
        hl = p2 - dvec_a / 3.0
    if h2 == "VECTOR":
        hr = p2 + dvec_b / 3.0
    if "FREE" not in (h1, h2) and "ALIGNED" in (h1, h2):
        la, lb = float(np.linalg.norm(p2 - hl)) or 1.0, float(np.linalg.norm(p2 - hr)) or 1.0
        ratio = la / lb
        sel = True                                     # f1 & SELECT (keys exported unselected: never ALIGNED here)
        if sel:
            if h2 == "ALIGNED" and la > 1e-5:
                hr = p2 + (1.0 / ratio) * (p2 - hl)
            if h1 == "ALIGNED" and lb > 1e-5:
                hl = p2 + ratio * (p2 - hr)
    b["handle_left"], b["handle_right"] = [float(hl[0]), float(hl[1])], [float(hr[0]), float(hr[1])]


def _is_autoh(b):
    return b["handle_left_type"] in AUTO and b["handle_right_type"] in AUTO


# ------------------------------------------------------------------ handles: smoothing pass (curve.cc)
def _clamp(hmax, hmin, i, dy, no_reverse, no_overshoot):
    if dy > 0:
        if no_overshoot:
            hmax[i] = min(hmax[i], dy)
        if no_reverse:
            hmin[i] = 0.0
    elif dy < 0:
        if no_reverse:
            hmax[i] = 0.0
        if no_overshoot:
            hmin[i] = max(hmin[i], dy)
    elif no_reverse or no_overshoot:
        hmax[i] = hmin[i] = 0.0


def _lock(a, b, c, d, i, value):
    a[i] = c[i] = 0.0
    b[i] = 1.0
    d[i] = value


def _tridiagonal(a, b, c, d, n):
    """BLI_tridiagonal_solve (double intermediates); non-cyclic (a[0] = c[n-1] = 0 here)."""
    c1, d1 = [0.0] * n, [0.0] * n
    c1[0] = c_prev = c[0] / b[0]
    d1[0] = d_prev = d[0] / b[0]
    for i in range(1, n):
        den = b[i] - a[i] * c_prev
        c1[i] = c_prev = c[i] / den
        d1[i] = d_prev = (d[i] - a[i] * d_prev) / den
    x = [0.0] * n
    x[n - 1] = x_prev = d_prev
    for i in range(n - 2, -1, -1):
        x_prev = d1[i] - c1[i] * x_prev
        x[i] = x_prev
    return x if math.isfinite(x_prev) else None


def _solve_with_limits(a, b, c, d, hmin, hmax, n):
    """tridiagonal_solve_with_limits: lock handles that overshoot their clamp range, unlock (at most twice) those that
    then want to move back into it."""
    a0, b0, c0, d0 = list(a), list(b), list(c), list(d)
    locked_flag, unlocks = [0] * n, [0] * n
    while True:
        h = _tridiagonal(a, b, c, d, n)
        if h is None:
            return None
        allx = locked = False
        overshoot = unlocked = False
        while True:
            for i in range(n):
                if hmin[i] <= h[i] <= hmax[i]:
                    continue
                overshoot = True
                target = hmax[i] if h[i] > hmax[i] else hmin[i]
                if target != 0.0 or allx:
                    locked_flag[i] = 1
                    _lock(a, b, c, d, i, target)
                    locked = True
            allx = True
            if not (overshoot and not locked):
                break
        if not locked:
            for i in range(n):
                if not locked_flag[i] or unlocks[i] >= 2:
                    continue
                state = a0[i] * h[(i + n - 1) % n] + b0[i] * h[i] + c0[i] * h[(i + 1) % n] - d0[i]
                relax = -state * b0[i]
                if (relax > 0 and h[i] < hmax[i]) or (relax < 0 and h[i] > hmin[i]):
                    a[i], b[i], c[i], d[i] = a0[i], b0[i], c0[i], d0[i]
                    locked_flag[i] = 0
                    unlocks[i] += 1
                    unlocked = True
        if not (overshoot or unlocked):
            return h


def _output_handle(bz, right, dy, endpoint):
    """bezier_output_handle(_inner): only Auto / Vector sides are written; an end point mirrors to its other side."""
    idx, oth = ("handle_right", "handle_left") if right else ("handle_left", "handle_right")
    hr = bz["handle_right_type"] if right else bz["handle_left_type"]
    hm = bz["handle_left_type"] if right else bz["handle_right_type"]
    if hr not in ("AUTO", "AUTO_CLAMPED", "VECTOR"):
        return
    new = [bz[idx][0], bz["co"][1] + dy]
    bz[idx] = new
    p = np.array(bz["co"], float)
    if hm == "ALIGNED":
        hlen = float(np.linalg.norm(p - np.array(bz[oth], float)))
        h2len = float(np.linalg.norm(p - np.array(new, float)))
        t = p - np.array(new, float)
        bz[oth] = list(p + t * (hlen / h2len))
    elif endpoint and hm in ("AUTO", "AUTO_CLAMPED", "VECTOR"):
        t = p - np.array(new, float)
        bz[oth] = list(p + t)


def _calc_handle_adj(hsize, dx):
    fac = dx / (hsize[0] + dx / 3.0)
    if fac < 1.0:
        hsize[0] *= fac
        hsize[1] *= fac
    return 1.0 - 3.0 * hsize[0] / dx


def _smooth_segment(bz, total, start, count):
    """bezier_handle_calc_smooth_fcurve, non-cyclic."""
    if count < 2:
        return
    first, last = bz[start], bz[start + count - 1]

    def solve_end(k, htype, end):
        return htype == "VECTOR" or (end and htype in AUTO and k["_auto"] == "NORMAL")

    solve_first = solve_end(first, first["handle_right_type"], start == 0)
    solve_last = solve_end(last, last["handle_left_type"], start + count == total)
    if count == 2 and solve_first == solve_last:
        return
    nan = float("nan")
    dx, dy = [nan] * count, [nan] * count
    for i in range(1, count):
        j = start + i
        dx[i] = bz[j]["co"][0] - bz[j - 1]["co"][0]
        dy[i] = bz[j]["co"][1] - bz[j - 1]["co"][1]
    lr = [1.0] * count
    for i in range(1, count - 1):
        lr[i] = dx[i + 1] / dx[i]
    hmax, hmin = [FLT_MAX] * count, [-FLT_MAX] * count
    clamped_cur = "AUTO_CLAMPED" in (first["handle_left_type"], first["handle_right_type"])
    for i in range(1, count):
        j = start + i
        clamped_prev = clamped_cur
        clamped_cur = "AUTO_CLAMPED" in (bz[j]["handle_left_type"], bz[j]["handle_right_type"])
        _clamp(hmax, hmin, i - 1, dy[i], clamped_prev, clamped_prev)
        _clamp(hmax, hmin, i, dy[i] * lr[i], clamped_cur, clamped_cur)
    a, b, c, d = [0.0] * count, [0.0] * count, [0.0] * count, [0.0] * count
    first_adj = last_adj = 0.0
    if not solve_first:
        tmp = [first["handle_right"][0] - first["co"][0], first["handle_right"][1] - first["co"][1]]
        first_adj = _calc_handle_adj(tmp, dx[1])
        _lock(a, b, c, d, 0, tmp[1])
    else:                                              # bezier_eq_noaccel_right
        a[0], b[0], c[0], d[0] = 0.0, 2.0, 1.0 / lr[1], dy[1]
    if not solve_last:
        tmp = [last["co"][0] - last["handle_left"][0], last["co"][1] - last["handle_left"][1]]
        last_adj = _calc_handle_adj(tmp, dx[count - 1])
        _lock(a, b, c, d, count - 1, tmp[1])
    else:                                              # bezier_eq_noaccel_left
        i = count - 1
        a[i], b[i], c[i], d[i] = lr[i] * lr[i], 2.0 * lr[i], 0.0, dy[i] * lr[i] * lr[i]
    for i in range(1, count - 1):                      # bezier_eq_continuous
        a[i], b[i], c[i], d[i] = lr[i] * lr[i], 2.0 * (lr[i] + 1), 1.0 / lr[i + 1], dy[i] * lr[i] * lr[i] + dy[i + 1]
    if count > 2 or solve_last:
        b[1] += lr[1] * first_adj
    if count > 2 or solve_first:
        b[count - 2] += last_adj
    h = _solve_with_limits(a, b, c, d, hmin, hmax, count)
    if h is None:
        return
    for i in range(1, count - 1):
        j = start + i
        end = j == total - 1
        _output_handle(bz[j], False, -h[i] / lr[i], end)
        _output_handle(bz[j], True, h[i], end)
    if solve_first:
        _output_handle(first, True, h[0], start == 0)
    if solve_last:
        _output_handle(last, False, -h[count - 1] / lr[count - 1], start + count == total)


def recalc_handles(keys, extrapolation="CONSTANT", smoothing="NONE"):
    """BKE_fcurve_handles_recalc_ex (non-cyclic curve), in place on the key dicts."""
    n = len(keys)
    if n < 2:
        return
    for idx, bz in enumerate(keys):
        prev = keys[idx - 1] if idx > 0 else None
        nxt = keys[idx + 1] if idx < n - 1 else None
        if bz["handle_left"][0] > bz["co"][0]:
            bz["handle_left"][0] = bz["co"][0]
        if bz["handle_right"][0] < bz["co"][0]:
            bz["handle_right"][0] = bz["co"][0]
        _calc_handle(bz, prev, nxt, smoothing)
        if _is_autoh(bz) and idx in (0, n - 1) and extrapolation == "CONSTANT":
            bz["handle_left"][1] = bz["handle_right"][1] = bz["co"][1]
            bz["_auto"] = "LOCKED"
        if prev is not None and prev["co"][0] >= bz["co"][0]:
            prev["_auto"] = bz["_auto"] = "LOCKED"
    if smoothing != "NONE":                             # BKE_nurb_handle_smooth_fcurve
        free = [(_is_autoh(k) and k["_auto"] == "NORMAL") for k in keys]
        start, count = 0, 1
        for j in range(1, n):
            if not free[j]:
                _smooth_segment(keys, n, start, count + 1)
                start, count = j, 1
            else:
                count += 1
        if count > 1:
            _smooth_segment(keys, n, start, count)
    for k in keys:
        k.pop("_auto", None)
