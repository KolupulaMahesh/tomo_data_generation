"""Numerical invariants of the common regional clock, independent of Blender."""
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

root = Path(__file__).resolve().parents[1] / 'common'
package = types.ModuleType('regional_test_core')
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location('regional_test_core.regional_retime', root / 'regional_retime.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_shared_clock_satisfies_both_joint_speed_limits():
    frames = np.arange(401) / 4
    q1, q2 = frames / 24, -frames / 12
    new, slopes = core.plan_retime(frames, [q1, q2],
        [dict(velocity=.5, acceleration=1), dict(velocity=.6, acceleration=1)], 24)
    assert np.all(np.diff(new) > 0)
    assert np.all((slopes > 0) & (slopes <= 1))
    assert np.allclose(np.diff(q1) / (np.diff(new)/24), .27)
    assert np.allclose(np.diff(q2) / (np.diff(new)/24), -.54)
    assert np.allclose(slopes, .27)


def test_local_motion_does_not_stretch_distant_safe_region():
    frames = np.arange(1201) / 4
    t = frames / 24
    u = np.clip(t/.5, 0, 1)
    q = 10*u**3 - 15*u**4 + 6*u**5
    new, slopes = core.plan_retime(frames, [q], [dict(velocity=1, acceleration=4)], 24)
    assert new[-1] > frames[-1]
    assert np.max(new - frames) > 1
    assert np.allclose(np.diff(new[-100:]), np.diff(frames[-100:]))
    assert np.allclose(slopes[-100:], 1)


def test_envelope_never_reduces_required_stretch():
    required = np.ones(100)
    required[20] = 4
    required[40:45] = 2
    result = core.smooth_envelope(required, 8)
    assert np.all(result >= required)
    assert np.all(result[60:] == 1)
    assert result[19] > 1 and result[21] > 1


def test_zero_limit_cannot_be_fixed_by_finite_stretch():
    frames = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match='zero velocity'):
        core.plan_retime(frames, [frames], [dict(velocity=0, acceleration=10)], 24)
    new, slopes = core.plan_retime(frames, [np.zeros(10)], [dict(velocity=0, acceleration=0)], 24)
    assert np.allclose(new, frames) and np.allclose(slopes, 1)


@pytest.mark.parametrize('bounds', [dict(velocity=None, acceleration=1), dict(velocity=1, acceleration=-1)])
def test_missing_or_invalid_limits_rejected(bounds):
    with pytest.raises(ValueError):
        core.plan_retime(np.arange(10), [np.arange(10)], [bounds], 24)


def test_nonuniform_grid_rejected():
    with pytest.raises(ValueError):
        core.plan_retime([0, 1, 3, 4], [[0, 1, 2, 3]], [dict(velocity=1, acceleration=1)], 24)


def test_skipped_jump_keeps_duration_while_other_motion_stretches():
    frames = np.arange(401) / 4
    u = np.clip(frames / 12, 0, 1)
    q = 10*u**3 - 15*u**4 + 6*u**5
    q[frames >= 80] += .5
    excluded = (frames >= 70) & (frames <= 90)
    new, slopes = core.plan_retime(frames, [q], [dict(velocity=1, acceleration=4)], 24, excluded=excluded)
    assert new[-1] > frames[-1]
    assert np.allclose(np.diff(new[excluded]), np.diff(frames[excluded]))
    assert np.allclose(slopes[excluded], 1)


def test_speedup_preserves_pauses_skipped_blocks_and_limits():
    spec = importlib.util.spec_from_file_location('regional_test_core.keyframe_retime', root / 'keyframe_retime.py')
    clock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clock)
    frames = np.arange(31, dtype=float)
    limits = [dict(velocity=10., acceleration=10.)]
    target = clock.accelerate_blocks(np.array([0.,10.,20.,30.]), np.array([0.,10.,20.,30.]),
        frames, [frames*.01], limits, 24., .9, 2., [(0,10),(20,30)], [(25,26)])
    assert np.allclose(target, [0,5,15,25])
    # A limiting second joint prevents compression even when the first is slow.
    target = clock.accelerate_blocks(np.array([0.,30.]), np.array([0.,60.]), frames,
        [frames*.01, frames], limits + [dict(velocity=12., acceleration=10.)],
        24., .9, 2., [(0,30)])
    assert np.allclose(target, [0,60])
