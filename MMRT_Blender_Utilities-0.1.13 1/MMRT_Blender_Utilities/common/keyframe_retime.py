"""Piecewise-affine shared timing for moving existing keys without resampling."""
import numpy as np


def shared_key_clock(old, proposed, intervals):
    """Merge overlapping motion segments so each remains affine in time.

    Any active curve crossing a boundary forces that boundary to disappear.
    This preserves interpolation and coordinated motion without inserting keys.
    """
    old, proposed = np.asarray(old, dtype=float), np.asarray(proposed, dtype=float)
    rates = np.diff(proposed) / np.diff(old)
    merged = []
    for lo, hi in sorted(intervals):
        lo, hi = max(float(old[0]), float(lo)), min(float(old[-1]), float(hi))
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1] + 1e-7:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    knots = sorted({float(old[0]), float(old[-1]), *(x for span in merged for x in span)})
    target = [knots[0]]
    for lo, hi in zip(knots, knots[1:]):
        motion = any(a <= lo+1e-7 and hi <= b+1e-7 for a, b in merged)
        selected = rates[(old[:-1] < hi) & (old[1:] > lo)]
        scale = max(1.0, float(selected.max())) if motion and len(selected) else 1.0
        target.append(target[-1] + (hi-lo)*scale)
    return np.asarray(knots), np.asarray(target)


def map_frame(frame, source, target):
    if frame < source[0]:
        return float(target[0] + frame-source[0])
    if frame > source[-1]:
        return float(target[-1] + frame-source[-1])
    return float(np.interp(frame, source, target))


def accelerate_blocks(source, target, frames, trajectories, limits, fps, margin,
                      max_speedup, intervals, skipped=()):
    """Shorten affine motion blocks using joint derivative headroom; caller validates."""
    from .joint_limits import motion_derivatives
    required = np.zeros(len(frames))
    dt = (frames[1]-frames[0])/fps
    for values, bounds in zip(trajectories, limits):
        derivatives = motion_derivatives(np.unwrap(values), dt)
        for metric, power in [('velocity', 1.), ('acceleration', .5)]:
            measured = np.abs(derivatives[metric])
            bound = bounds[metric]
            if bound is None or not np.isfinite(bound) or bound < 0:
                raise ValueError('Speedup requires finite velocity and acceleration limits.')
            demand = np.where(measured > 1e-6, np.inf, 0.) if bound == 0 else (measured/(bound*margin))**power
            required = np.maximum(required, demand)
    result = [target[0]]
    for i, (lo, hi) in enumerate(zip(source, source[1:])):
        scale = (target[i+1]-target[i])/(hi-lo)
        # Keep pauses and unrepairable transitions; include boundary stencils.
        motion = any(a < hi and b > lo for a, b in intervals)
        protected = any(a <= hi and b >= lo for a, b in skipped)
        mask = (frames >= lo-2*(frames[1]-frames[0])) & (frames <= hi+2*(frames[1]-frames[0]))
        need = float(required[mask].max()) if mask.any() else np.inf
        if motion and not protected and need < 1:
            scale = max(1/max_speedup, need)
        result.append(result[-1]+(hi-lo)*scale)
    return np.asarray(result)


def speedup_violations(issues, source, target, output_ratio=1.):
    """Find violations in compressed blocks plus a two-output-frame stencil guard.

    Position violations remain unconditional failures. Use unfiltered issues so
    skipped discontinuities beside an accelerated region cannot hide a failure.
    """
    source, target = np.asarray(source), np.asarray(target)
    guard = 2. / output_ratio
    regions = [(lo-guard, hi+guard) for lo, hi, original_duration in
               zip(target, target[1:], np.diff(source))
               if hi-lo < original_duration-1e-7]
    failures = []
    for issue in issues:
        frame = map_frame(issue['source_frame'], source, target)
        if issue['metric'] == 'position' or any(lo <= frame <= hi for lo, hi in regions):
            failures.append(issue)
    return regions, failures
