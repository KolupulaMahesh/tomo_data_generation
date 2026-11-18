"""Dependency-light joint-limit parsing and sampled trajectory checks (SI units)."""
import ast
import json
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


def parse_overrides(text):
    """Read JSON or MoveIt's scalar-mapping joint_limits.yaml subset.

    No YAML dependency is required inside Blender. Complex YAML constructs are
    deliberately rejected instead of being interpreted differently from YAML.
    """
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
        stack = [(-1, data)]
        for number, line in enumerate(text.splitlines(), 1):
            if '\t' in line:
                raise ValueError(f'Limits line {number}: use spaces, not tabs.')
            line = line.split('#', 1)[0].rstrip()
            if not line.strip() or line.strip() in ('---', '...'):
                continue
            match = re.fullmatch(r'( *)([^:]+):\s*(.*)', line)
            if not match:
                raise ValueError(f'Limits line {number}: expected a scalar mapping; complex YAML is unsupported.')
            spaces, key, value = match.groups()
            key = key.strip()
            if key.startswith(('"', "'")):
                key = ast.literal_eval(key)
            if not isinstance(key, str) or re.search(r'[\[\]{}&*!>|]', key):
                raise ValueError(f'Limits line {number}: unsupported key.')
            indent = len(spaces)
            while stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]
            if key in parent:
                raise ValueError(f'Limits line {number}: duplicate key {key}.')
            if not value:
                parent[key] = {}
                stack.append((indent, parent[key]))
            elif value.lower() in ('true', 'false'):
                parent[key] = value.lower() == 'true'
            else:
                try:
                    parent[key] = float(value)
                except ValueError as exc:
                    raise ValueError(f'Limits line {number}: expected a number or true/false.') from exc
    if not isinstance(data, dict) or not isinstance(data.get('joint_limits'), dict):
        raise ValueError('Limits must contain a joint_limits mapping.')
    for name, values in data['joint_limits'].items():
        if not isinstance(name, str) or not isinstance(values, dict):
            raise ValueError('Each joint_limits entry must be a named mapping.')
    return data['joint_limits']


def limits_from_urdf(text, overrides):
    root = ET.fromstring(text)
    if root.tag != 'robot':
        raise ValueError('Expected a URDF <robot> document.')
    limits = {}
    for joint in root.findall('joint'):
        name, kind = joint.get('name'), joint.get('type')
        if not name or name in limits:
            raise ValueError('URDF joint names must be present and unique.')
        raw = joint.find('limit')
        raw = raw.attrib if raw is not None else {}
        extra = overrides.get(name, {})
        entry = {'mimic': joint.find('mimic') is not None, 'type': kind, 'lower': None, 'upper': None, 'velocity': None, 'acceleration': None}
        for key, default in [('has_position_limits', kind != 'continuous'),
                             ('has_velocity_limits', True), ('has_acceleration_limits', True)]:
            if not isinstance(extra.get(key, default), bool):
                raise ValueError(f'{name}: {key} must be true or false.')
        if kind != 'continuous' and extra.get('has_position_limits', True):
            entry['lower'] = extra.get('min_position', raw.get('lower'))
            entry['upper'] = extra.get('max_position', raw.get('upper'))
        if extra.get('has_velocity_limits', True):
            entry['velocity'] = extra.get('max_velocity', raw.get('velocity'))
        if extra.get('has_acceleration_limits', True):
            entry['acceleration'] = extra.get('max_acceleration', raw.get('acceleration'))
        for key in ('lower', 'upper', 'velocity', 'acceleration'):
            value = entry[key]
            if value is not None:
                if isinstance(value, bool):
                    raise ValueError(f'{name}: {key} must be numeric.')
                value = float(value)
                if not math.isfinite(value) or (key in ('velocity', 'acceleration') and value < 0):
                    raise ValueError(f'{name}: invalid {key} limit.')
                entry[key] = value
        if entry['lower'] is not None and entry['upper'] is not None and entry['lower'] > entry['upper']:
            raise ValueError(f'{name}: lower position limit exceeds upper limit.')
        limits[name] = entry
    if not limits:
        raise ValueError('URDF has no joints.')
    return limits


def motion_derivatives(positions, dt):
    """SI derivatives on a uniform time grid; shared by checks and retiming."""
    positions = np.asarray(positions, dtype=float)
    values = {'position': positions}
    if len(positions) >= 2:
        values['velocity'] = np.gradient(positions, dt, edge_order=2 if len(positions) >= 3 else 1)
    if len(positions) >= 3:
        acceleration = np.empty_like(positions)
        acceleration[1:-1] = (positions[2:] - 2 * positions[1:-1] + positions[:-2]) / dt**2
        if len(positions) >= 4:
            acceleration[0] = (2*positions[0] - 5*positions[1] + 4*positions[2] - positions[3]) / dt**2
            acceleration[-1] = (2*positions[-1] - 5*positions[-2] + 4*positions[-3] - positions[-4]) / dt**2
        else:
            acceleration[0] = acceleration[-1] = acceleration[1]
        values['acceleration'] = acceleration
    return values


def analyze_samples(frames, positions, fps, limits, *, angular=False, tolerance=1e-6):
    """Return (violations, unchecked checks) for uniformly spaced output frames.

    Derivatives use elapsed OUTPUT time, including fractional FPS. Endpoints use
    one-sided stencils; acceleration requires at least three samples. Orientation
    samples are unwrapped before differentiation to avoid +/-pi branch spikes.
    """
    frames = np.asarray(frames, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if frames.ndim != 1 or positions.shape != frames.shape or len(frames) == 0:
        raise ValueError('Expected matching nonempty frame and position arrays.')
    if not np.isfinite(positions).all() or not np.isfinite(frames).all():
        raise ValueError('Animation produced nonfinite frame or joint values.')
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('FPS must be positive and tolerance nonnegative.')
    if len(frames) > 1 and (np.any(np.diff(frames) <= 0) or not np.allclose(np.diff(frames), frames[1] - frames[0])):
        raise ValueError('Samples must be increasing and uniformly spaced.')
    if angular:
        positions = np.unwrap(positions)
    dt = (frames[1] - frames[0]) / fps if len(frames) > 1 else 1.0
    values = motion_derivatives(positions, dt)
    bounds = {
        'position': (limits['lower'], limits['upper']),
        'velocity': (None if limits['velocity'] is None else -limits['velocity'], limits['velocity']),
        'acceleration': (None if limits['acceleration'] is None else -limits['acceleration'], limits['acceleration']),
    }
    issues, unchecked = [], []
    for metric, (lower, upper) in bounds.items():
        if lower is None and upper is None:
            if metric != 'position' or limits['type'] != 'continuous':
                unchecked.append(f'{metric}: limit missing or disabled')
            continue
        if metric not in values:
            unchecked.append(f'{metric}: insufficient samples')
            continue
        if metric == 'position' and (lower is None or upper is None):
            unchecked.append('position: one bound missing')
        for frame, value in zip(frames, values[metric]):
            if (lower is not None and value < lower - tolerance) or (upper is not None and value > upper + tolerance):
                issues.append({'frame': float(frame), 'metric': metric, 'value': float(value),
                               'lower': lower, 'upper': upper})
    return issues, unchecked
