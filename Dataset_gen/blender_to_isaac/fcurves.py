"""Blender F-curve evaluation (Blender 5.2, source/blender/blenkernel/intern/fcurve.cc), pure Python, for the packing
rig exported by blender_scripts/export_rig.py. Covers what the rig uses: BEZIER (stored handles),
CONSTANT, LINEAR and the easing interpolations, CONSTANT / LINEAR extrapolation. No F-curve modifiers (the rig has none).
"""
from __future__ import annotations

import math

import numpy as np

SMALL = -1.0e-10
FLT_EPSILON = 1.1920929e-07
BINARY_THRESH = 0.0001                     # BKE_fcurve_bezt_binarysearch_index_ex threshold used for evaluation
f32 = np.float32


def _sqrt3d(d):
    if d == 0.0:
        return 0.0
    if d < 0:
        return -math.exp(math.log(-d) / 3.0)
    return math.exp(math.log(d) / 3.0)


def _ok(x):
    return SMALL <= x <= 1.000001


def _solve_cubic(c0, c1, c2, c3):
    """Blender's solve_cubic: the roots in [0, 1] (as float32), in Blender's order."""
    o = []
    if c3 != 0.0:
        a, b, c = c2 / c3, c1 / c3, c0 / c3
        a = a / 3
        p = b / 3 - a * a
        q = (2 * a * a * a - a * b + c) / 2
        d = q * q + p * p * p
        if d > 0.0:
            t = math.sqrt(d)
            r = float(f32(_sqrt3d(-q + t) + _sqrt3d(-q - t) - a))
            return [r] if _ok(r) else []
        if d == 0.0:
            t = _sqrt3d(-q)
            for r in (float(f32(2 * t - a)), float(f32(-t - a))):
                if _ok(r):
                    o.append(r)
            return o
        phi = math.acos(-q / math.sqrt(-(p * p * p)))
        t = math.sqrt(-p)
        p = math.cos(phi / 3)
        q = math.sqrt(3 - 3 * p * p)
        for r in (float(f32(2 * t * p - a)), float(f32(-t * (p + q) - a)), float(f32(-t * (p - q) - a))):
            if _ok(r):
                o.append(r)
        return o
    a, b, c = c2, c1, c0
    if a != 0.0:
        p = b * b - 4 * a * c
        if p > 0:
            p = math.sqrt(p)
            for r in (float(f32((-b - p) / (2 * a))), float(f32((-b + p) / (2 * a)))):
                if _ok(r):
                    o.append(r)
            return o
        if p == 0:
            r = float(f32(-b / (2 * a)))
            return [r] if _ok(r) else []
        return []
    if b != 0.0:
        r = float(f32(-c / b))
        return [r] if _ok(r) else []
    return [0.0] if c == 0.0 else []


def _findzero(x, q0, q1, q2, q3):
    c0 = float(f32(q0) - f32(x))
    c1 = float(f32(3.0) * (f32(q1) - f32(q0)))
    c2 = float(f32(3.0) * (f32(q0) - f32(2.0) * f32(q1) + f32(q2)))
    c3 = float(f32(q3) - f32(q0) + f32(3.0) * (f32(q1) - f32(q2)))
    return _solve_cubic(c0, c1, c2, c3)


def _berekeny(f1, f2, f3, f4, t):
    f1, f2, f3, f4, t = f32(f1), f32(f2), f32(f3), f32(f4), f32(t)
    c0 = f1
    c1 = f32(3.0) * (f2 - f1)
    c2 = f32(3.0) * (f1 - f32(2.0) * f2 + f3)
    c3 = f4 - f1 + f32(3.0) * (f2 - f3)
    return float(c0 + t * c1 + t * t * c2 + t * t * t * c3)


def _correct_bezpart(v1, v2, v3, v4):
    """BKE_fcurve_correct_bezpart: pull the handles in so the segment cannot loop back in time."""
    h1 = [v1[0] - v2[0], v1[1] - v2[1]]
    h2 = [v4[0] - v3[0], v4[1] - v3[1]]
    ln = v4[0] - v1[0]
    len1, len2 = abs(h1[0]), abs(h2[0])
    if len1 + len2 == 0.0:
        return v2, v3
    if len1 + len2 > ln:
        fac = ln / (len1 + len2)
        v2 = [v1[0] - fac * h1[0], v1[1] - fac * h1[1]]
        v3 = [v4[0] - fac * h2[0], v4[1] - fac * h2[1]]
    return v2, v3


# Robert Penner easing (BLI_easing.cc), time / begin / change / duration
def _ease(kind, mode, t, b, c, d, back=1.70158, amplitude=0.0, period=0.0):
    def ein(x):
        return {"QUAD": x * x, "CUBIC": x ** 3, "QUART": x ** 4, "QUINT": x ** 5,
                "SINE": 1 - math.cos(x * math.pi / 2), "EXPO": 0.0 if x == 0 else 2 ** (10 * (x - 1)),
                "CIRC": 1 - math.sqrt(max(0.0, 1 - x * x))}[kind]
    x = t / d
    if kind not in ("QUAD", "CUBIC", "QUART", "QUINT", "SINE", "EXPO", "CIRC"):
        raise NotImplementedError(f"easing interpolation {kind} (the rig has none)")
    if mode in ("AUTO", "EASE_IN"):           # AUTO = ease in for the transitional types (fcurve.cc)
        return c * ein(x) + b
    if mode == "EASE_OUT":
        return c * (1 - ein(1 - x)) + b
    if x < 0.5:                                # EASE_IN_OUT
        return c * ein(2 * x) / 2 + b
    return c * (1 - ein(2 - 2 * x) / 2) + b


class FCurve:
    def __init__(self, d: dict):
        self.data_path, self.index = d["data_path"], d["array_index"]
        self.extrapolation = d["extrapolation"]
        if d.get("modifiers"):
            raise NotImplementedError(f"F-curve modifiers on {self.data_path} (the rig has none)")
        self.keys = d["keys"]
        self.x = [k["co"][0] for k in self.keys]

    def __call__(self, t: float) -> float:
        k, n = self.keys, len(self.keys)
        if n == 0:
            return 0.0
        if t <= self.x[0]:
            return self._extrapolate(t, 0, +1)
        if self.x[-1] <= t:
            return self._extrapolate(t, n - 1, -1)
        a = int(np.searchsorted(self.x, t))      # first key >= t
        for j in (a - 1, a):                      # exact hit within the search threshold: that key's value
            if 0 <= j < n and abs(self.x[j] - t) < BINARY_THRESH:
                return float(f32(k[j]["co"][1]))
        prev, nxt = k[a - 1], k[a]
        ipo = prev["interpolation"]
        begin, change = prev["co"][1], nxt["co"][1] - prev["co"][1]
        duration, time = nxt["co"][0] - prev["co"][0], t - prev["co"][0]
        if ipo == "CONSTANT" or duration == 0:
            return float(f32(begin))
        if ipo == "LINEAR":
            return float(f32(change * time / duration + begin))
        if ipo == "BEZIER":
            v1, v2 = list(prev["co"]), list(prev["handle_right"])
            v3, v4 = list(nxt["handle_left"]), list(nxt["co"])
            if (abs(v1[1] - v4[1]) < FLT_EPSILON and abs(v2[1] - v3[1]) < FLT_EPSILON
                    and abs(v3[1] - v4[1]) < FLT_EPSILON):
                return float(f32(v1[1]))
            v2, v3 = _correct_bezpart(v1, v2, v3, v4)
            roots = _findzero(t, v1[0], v2[0], v3[0], v4[0])
            if not roots:
                return 0.0
            return _berekeny(v1[1], v2[1], v3[1], v4[1], roots[0])
        return float(f32(_ease(ipo, prev["easing"], time, begin, change, duration, prev["back"], prev["amplitude"],
                               prev["period"])))

    def _extrapolate(self, t, i, direction):
        k = self.keys[i]
        if self.extrapolation == "CONSTANT" or k["interpolation"] == "CONSTANT" or len(self.keys) == 1:
            return float(f32(k["co"][1]))
        # LINEAR extrapolation: on along the end key's outer handle (Bezier), else along the end segment
        if k["interpolation"] == "BEZIER":
            h = k["handle_left"] if direction > 0 else k["handle_right"]
            dx = k["co"][0] - h[0]
            slope = (k["co"][1] - h[1]) / dx if dx else 0.0
        else:
            o = self.keys[i + direction]
            dx = k["co"][0] - o["co"][0]
            slope = (k["co"][1] - o["co"][1]) / dx if dx else 0.0
        return float(f32(k["co"][1] + slope * (t - k["co"][0])))
