"""Numerical checks without importing Blender or requiring a robot simulator."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('joint_limits_core', Path(__file__).resolve().parents[1] / 'common/joint_limits.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

URDF = '''<robot name="test"><joint name="j" type="revolute"><limit lower="-2" upper="2" velocity="3"/></joint></robot>'''


def limits(**kwargs):
    result = dict(type='revolute', lower=-100, upper=100, velocity=100, acceleration=100)
    result.update(kwargs)
    return result


def test_tomo_yaml_and_urdf():
    path = Path(__file__).resolve().parents[3] / 'TomO_files/tomo_re_desc/config/joint_limits.yaml'
    parsed = core.parse_overrides(path.read_text())
    assert parsed['Left_arm_joint1']['max_acceleration'] == 8.7263
    config = core.parse_overrides('joint_limits:\n  j:\n    has_velocity_limits: true\n    max_velocity: 1\n    has_acceleration_limits: true\n    max_acceleration: 4\n')
    actual = core.limits_from_urdf(URDF, config)['j']
    assert actual == dict(mimic=False, type='revolute', lower=-2, upper=2, velocity=1, acceleration=4)


@pytest.mark.parametrize('text', [
    'joint_limits:\n  j:\n    max_velocity: nope',
    'joint_limits:\n  j:\n    max_velocity: 1\n    max_velocity: 2',
    'joint_limits: []',
    '{"joint_limits": {"j": []}}',
])
def test_reject_malformed_limits(text):
    with pytest.raises(ValueError):
        core.parse_overrides(text)


@pytest.mark.parametrize('entry', [{'max_velocity': -1}, {'max_acceleration': float('nan')},
                                  {'min_position': 5, 'max_position': 1}, {'has_velocity_limits': 1}])
def test_invalid_bounds(entry):
    with pytest.raises(ValueError):
        core.limits_from_urdf(URDF, {'j': entry})


def test_negative_velocity_and_original_frame_numbers():
    frames = np.arange(100, 111, dtype=float)
    q = -2 * (frames - 100) / 10
    found, unchecked = core.analyze_samples(frames, q, 10, limits(velocity=1))
    assert not unchecked
    assert {v['metric'] for v in found} == {'velocity'}
    assert {v['frame'] for v in found} == set(frames)
    assert all(abs(v['value'] + 2) < 1e-10 for v in found)


def test_acceleration_quadratic_including_endpoints_fractional_fps():
    frames = np.arange(41) / 4 + 100
    fps = 30 / 1.001
    t = (frames - frames[0]) / fps
    found, _ = core.analyze_samples(frames, 3*t*t, fps, limits(acceleration=5))
    acceleration = [v for v in found if v['metric'] == 'acceleration']
    assert len(acceleration) == len(frames)
    assert np.allclose([v['value'] for v in acceleration], 6)


def test_stretch_reduces_speed_and_acceleration():
    frames = np.arange(21, dtype=float)
    q = (frames / 10)**2
    fast, _ = core.analyze_samples(frames, q, 10, limits(velocity=2.1, acceleration=1))
    slow, _ = core.analyze_samples(frames*2, q, 10, limits(velocity=2.1, acceleration=1))
    assert fast and not slow


def test_rotation_wrap_does_not_create_speed_spike():
    q = np.array([3.0, 3.1, 3.2, 3.3])
    wrapped = (q + np.pi) % (2*np.pi) - np.pi
    found, unchecked = core.analyze_samples(np.arange(4), wrapped, 10,
        limits(type='continuous', lower=None, upper=None, velocity=1.1, acceleration=0.1), angular=True)
    assert not found and not unchecked


def test_missing_limits_and_short_range_not_passed():
    found, unchecked = core.analyze_samples([5], [0], 24, limits(velocity=None, acceleration=1))
    assert not found
    assert 'velocity: limit missing or disabled' in unchecked
    assert 'acceleration: insufficient samples' in unchecked


def test_position_and_tolerance():
    found, _ = core.analyze_samples([1, 2, 3], [-1.000001, 0, 1.2], 1,
                                    limits(lower=-1, upper=1), tolerance=1e-5)
    assert len(found) == 1 and found[0]['frame'] == 3 and found[0]['metric'] == 'position'


def test_nonfinite_sample_rejected():
    with pytest.raises(ValueError):
        core.analyze_samples([1, 2, 3], [0, float('nan'), 1], 24, limits())
