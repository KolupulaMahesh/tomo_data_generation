"""Blender's matrix / rotation maths (blenlib math_rotation.cc, math_matrix.cc) in numpy, row-major 4x4
(translation in column 3), as mathutils returns them."""
from __future__ import annotations

import math

import numpy as np

# rotation orders: axes (i, j, k) and parity, as Blender's rotOrders table
ROT_ORDERS = {"XYZ": ((0, 1, 2), 0), "XZY": ((0, 2, 1), 1), "YXZ": ((1, 0, 2), 1),
              "YZX": ((1, 2, 0), 0), "ZXY": ((2, 0, 1), 0), "ZYX": ((2, 1, 0), 1)}
FLT_EPSILON = 1.1920929e-07


def eul_to_mat3(e, order="XYZ"):
    """eulO_to_mat3 (Blender writes M[col][row]; returned here row-major)."""
    (i, j, k), parity = ROT_ORDERS[order]
    ti, tj, th = (-e[i], -e[j], -e[k]) if parity else (e[i], e[j], e[k])
    ci, cj, ch = math.cos(ti), math.cos(tj), math.cos(th)
    si, sj, sh = math.sin(ti), math.sin(tj), math.sin(th)
    cc, cs, sc, ss = ci * ch, ci * sh, si * ch, si * sh
    M = np.zeros((3, 3))                       # M[col][row] as in Blender
    M[i][i], M[j][i], M[k][i] = cj * ch, sj * sc - cs, sj * cc + ss
    M[i][j], M[j][j], M[k][j] = cj * sh, sj * ss + cc, sj * cs - sc
    M[i][k], M[j][k], M[k][k] = -sj, cj * si, cj * ci
    return M.T


def mat3_normalized_to_eul(R, order="XYZ"):
    """mat3_normalized_to_eulO: of Blender's two Euler solutions, the one with the smaller sum of |angles|."""
    (i, j, k), parity = ROT_ORDERS[order]
    m = R.T                                    # Blender indexing m[col][row]
    cy = math.hypot(m[i][i], m[i][j])
    e1, e2 = [0.0] * 3, [0.0] * 3
    if cy > 16.0 * FLT_EPSILON:
        e1[i] = math.atan2(m[j][k], m[k][k])
        e1[j] = math.atan2(-m[i][k], cy)
        e1[k] = math.atan2(m[i][j], m[i][i])
        e2[i] = math.atan2(-m[j][k], -m[k][k])
        e2[j] = math.atan2(-m[i][k], -cy)
        e2[k] = math.atan2(-m[i][j], -m[i][i])
    else:
        e1[i] = math.atan2(-m[k][j], m[j][j])
        e1[j] = math.atan2(-m[i][k], cy)
        e1[k] = 0.0
        e2 = list(e1)
    if parity:
        e1, e2 = [-x for x in e1], [-x for x in e2]
    return e2 if sum(map(abs, e1)) > sum(map(abs, e2)) else e1


def normalize_cols(R):
    n = np.linalg.norm(R, axis=0)
    return R / np.where(n == 0, 1.0, n)


def mat_to_eul(M, order="XYZ"):
    """mat4_to_eulO / mat3_to_eulO: normalise the axes, then the best Euler."""
    return mat3_normalized_to_eul(normalize_cols(np.asarray(M)[:3, :3]), order)


def quat_normalize(q):
    q = np.asarray(q, float)
    n = math.sqrt(float(q @ q))
    return np.array([1.0, 0, 0, 0]) if n == 0 else q / n


def quat_to_mat3(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat3_to_quat(R):
    R = normalize_cols(R)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return quat_normalize(q)


def axis_angle_to_mat3(axis, angle):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) or 1.0)
    x, y, z = a
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def trs(loc, R3, size):
    M = np.eye(4)
    M[:3, :3] = R3 @ np.diag(size)
    M[:3, 3] = loc
    return M


def mat4_to_loc_rot_size(M):
    """mat4_to_loc_rot_size: rotation = normalised axes (negated for a negative matrix), size = inv(rot) @ mat3."""
    m3 = np.asarray(M)[:3, :3]
    rot = normalize_cols(m3)
    if np.linalg.det(m3) < 0:
        rot = -rot
    size = np.diag(np.linalg.inv(rot) @ m3).copy()
    return np.asarray(M)[:3, 3].copy(), rot, size


def mat4_to_size(M):
    return np.linalg.norm(np.asarray(M)[:3, :3], axis=0)


def interp_qt(a, b, t):
    cosom = float(np.dot(a, b))
    if cosom < 0:
        cosom, a = -cosom, -a
    if 1.0 - cosom > 0.0001:
        om = math.acos(min(1.0, cosom))
        so = math.sin(om)
        s1, s2 = math.sin((1 - t) * om) / so, math.sin(t * om) / so
    else:
        s1, s2 = 1 - t, t
    return s1 * a + s2 * b


def polar(A):
    U, S, Vt = np.linalg.svd(A)
    return U @ Vt, Vt.T @ np.diag(S) @ Vt


def interp_m4(A, B, t):
    """interp_m4_m4m4: location linear, rotation by polar decomposition + quaternion slerp, the rest linear."""
    UA, PA = polar(A[:3, :3])
    UB, PB = polar(B[:3, :3])
    if np.linalg.det(UA) < 0:
        UA, PA = -UA, -PA
    if np.linalg.det(UB) < 0:
        UB, PB = -UB, -PB
    U = quat_to_mat3(interp_qt(mat3_to_quat(UA), mat3_to_quat(UB), t))
    R = np.eye(4)
    R[:3, :3] = U @ (PA + (PB - PA) * t)
    R[:3, 3] = A[:3, 3] + (B[:3, 3] - A[:3, 3]) * t
    return R


def rotation_between_vecs(v1, v2):
    """rotation_between_vecs_to_mat3: the smallest rotation taking v1 onto v2."""
    a = np.asarray(v1, float) / (np.linalg.norm(v1) or 1.0)
    b = np.asarray(v2, float) / (np.linalg.norm(v2) or 1.0)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.clip(a @ b, -1, 1))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return axis_angle_to_mat3(perp, math.pi)
    return axis_angle_to_mat3(axis / s, math.atan2(s, c))
