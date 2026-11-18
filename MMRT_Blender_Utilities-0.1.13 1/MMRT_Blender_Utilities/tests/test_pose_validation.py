import importlib.util
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('poses', Path(__file__).resolve().parents[1]/'common/pose_validation.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def rotated(degrees):
    angle = np.radians(degrees)
    result = np.eye(4)
    result[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return result


def test_small_rotation_noise_accepted_but_real_rotation_rejected():
    assert core.poses_match(np.eye(4), rotated(.02))
    assert not core.poses_match(np.eye(4), rotated(.1))
    assert np.isclose(core.pose_errors(np.eye(4), rotated(.1))[1], .1)


def test_translation_scale_shear_and_reflection_still_rejected():
    for row, column, value in [(0,3,.0002), (0,0,1.001), (0,1,.001), (0,0,-1)]:
        changed = np.eye(4)
        changed[row,column] = value
        assert not core.poses_match(np.eye(4), changed)


def test_scaled_reflected_pose_and_batch():
    original = np.diag([-2.,3.,4.,1.])
    changed = rotated(.02) @ original
    assert core.poses_match(np.array([original, np.eye(4)]), np.array([changed, np.eye(4)]))
