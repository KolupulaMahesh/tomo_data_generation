"""Construct a shared, monotone regional clock from sampled joint trajectories."""
import numpy as np

from .joint_limits import motion_derivatives


def smooth_envelope(required, radius):
    """Upper envelope with localized C2 bump support; never reduces required scale."""
    extra = np.maximum(np.asarray(required, dtype=float) - 1.0, 0.0)
    result = extra.copy()
    radius = max(2, int(radius))
    for shift in range(1, min(radius, len(extra))):
        weight = (1 - (shift / radius)**2)**3
        result[shift:] = np.maximum(result[shift:], extra[:-shift] * weight)
        result[:-shift] = np.maximum(result[:-shift], extra[shift:] * weight)
    return result + 1.0


def plan_retime(frames, trajectories, limits, fps, *, margin=0.9, transition_seconds=0.25, excluded=None):
    """Return new output frames and du/dt slopes at the original sample frames.

    Both arrays use frame coordinates. All joints use the SAME clock. Local
    scaling includes the acceleration term caused by changing the clock rate.
    This is a proposal; Blender evaluates and validates the resulting curves.
    Excluded samples keep unit clock speed and do not contribute derivative
    constraints. The caller must report these regions as unchecked/unrepaired.
    """
    frames = np.asarray(frames, dtype=float)
    if len(frames) < 4 or not np.allclose(np.diff(frames), frames[1] - frames[0]) or np.any(np.diff(frames) <= 0):
        raise ValueError('Retiming needs at least four uniformly spaced samples.')
    if not 0 < margin < 1 or fps <= 0 or transition_seconds <= 0:
        raise ValueError('Invalid retiming margin, FPS, or transition duration.')
    if len(trajectories) != len(limits) or not trajectories:
        raise ValueError('Retiming needs trajectories and limits for every joint.')
    excluded = np.zeros(len(frames), dtype=bool) if excluded is None else np.asarray(excluded, dtype=bool)
    if excluded.shape != frames.shape:
        raise ValueError("Excluded samples must match frames.")
    dt = (frames[1] - frames[0]) / fps
    derivatives = []
    required = np.ones(len(frames))
    for values, bounds in zip(trajectories, limits):
        values = np.asarray(values, dtype=float)
        if values.shape != frames.shape or not np.isfinite(values).all():
            raise ValueError('Invalid trajectory samples.')
        values = np.unwrap(values)  # Caller supplies rotational recorder joints.
        data = motion_derivatives(values, dt)
        velocity, acceleration = data['velocity'], data['acceleration']
        velocity[excluded] = 0
        acceleration[excluded] = 0
        maximums = []
        for metric, measured in [('velocity', velocity), ('acceleration', acceleration)]:
            bound = bounds[metric]
            if bound is None or not np.isfinite(bound) or bound < 0:
                raise ValueError(f'Every joint needs a finite nonnegative {metric} limit.')
            if bound == 0:
                if np.max(np.abs(measured)) > 1e-6:
                    raise ValueError(f'Motion with a zero {metric} limit cannot be fixed by finite stretching.')
                bound = np.inf
            maximums.append(bound * margin)
        v_max, a_max = maximums
        required = np.maximum(required, np.abs(velocity) / v_max)
        required = np.maximum(required, np.sqrt(np.abs(acceleration) / a_max))
        derivatives.append((velocity, acceleration, v_max, a_max))
    radius = max(2, int(np.ceil(transition_seconds / dt)))
    scale = smooth_envelope(required, radius)
    scale[excluded] = 1
    for _ in range(40):
        # dt_new/dt_old = scale. d²q/dt_new² includes scale's derivative.
        scale_rate = np.gradient(scale, dt, edge_order=2)
        correction = np.ones(len(frames))
        for velocity, acceleration, v_max, a_max in derivatives:
            v_new = velocity / scale
            a_new = acceleration / scale**2 - velocity * scale_rate / scale**3
            correction = np.maximum(correction, np.abs(v_new) / v_max)
            correction = np.maximum(correction, np.sqrt(np.abs(a_new) / a_max))
        if np.max(correction) <= 1.0001:
            break
        scale = smooth_envelope(scale * np.where(correction > 1, correction * 1.02, 1), radius)
        scale[excluded] = 1
        if not np.isfinite(scale).all() or np.max(scale) > 100:
            raise ValueError('Required stretch exceeds 100x; inspect discontinuities and limits.')
    else:
        raise ValueError('Could not find a smooth regional timing proposal.')
    new = np.empty_like(frames)
    new[0] = frames[0]
    new[1:] = new[0] + np.cumsum((scale[:-1] + scale[1:]) * 0.5 * np.diff(frames))
    slopes = 1 / scale
    # Monotone Hermite handles: no reversal or overshoot between clock keys.
    secants = np.diff(frames) / np.diff(new)
    for i, secant in enumerate(secants):
        length = np.hypot(slopes[i] / secant, slopes[i + 1] / secant)
        if length > 3:
            slopes[i:i+2] *= 3 / length
    return new, slopes
