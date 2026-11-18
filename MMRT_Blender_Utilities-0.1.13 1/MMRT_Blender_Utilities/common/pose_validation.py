"""Compare evaluated poses with separate translation, rotation and stretch units."""
import numpy as np


def pose_errors(expected, actual):
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    translation = np.linalg.norm(expected[..., :3, 3] - actual[..., :3, 3], axis=-1)
    def polar(matrix):
        u, singular, vh = np.linalg.svd(matrix[..., :3, :3])
        orientation = u @ vh
        stretch = np.swapaxes(vh, -1, -2) @ (singular[..., :, None] * vh)
        return orientation, stretch
    left, left_stretch = polar(expected)
    right, right_stretch = polar(actual)
    relative = np.swapaxes(left, -1, -2) @ right
    angle = np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))
    angle = np.where(np.linalg.det(relative) < 0, 180., angle)
    stretch = np.max(np.abs(left_stretch-right_stretch), axis=(-2, -1))
    return translation, angle, stretch


def poses_match(expected, actual):
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        return False
    translation, angle, stretch = pose_errors(expected, actual)
    return bool(np.all(translation <= 1e-4) and np.all(angle <= .05) and np.all(stretch <= 1e-4))
