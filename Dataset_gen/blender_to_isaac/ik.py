"""Blender's LEGACY IK solver (intern/iksolver: IK_QJacobianSolver, IK_QJacobian::InvertSDLS, IK_QSegment, IK_QTask)
for one chain, as the packing rig uses it: revolute (one-axis) and null (locked) segments, a position goal and / or an
orientation goal on the tip, joint limits, stiffness. Units: the chain root's frame (iksolver_plugin builds it).
"""
from __future__ import annotations

import math

import numpy as np

IK_EPSILON = 1e-20
MAX_ANGLE_CHANGE = math.pi / 4


def rot_axis(angle, axis):
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_angle_from_matrix(R, axis):
    t = math.sqrt(R[0, 0] ** 2 + R[0, 1] ** 2)
    if t > 16.0 * IK_EPSILON:
        if axis == 0:
            return -math.atan2(R[1, 2], R[2, 2])
        if axis == 1:
            return math.atan2(-R[0, 2], t)
        return -math.atan2(R[0, 1], R[0, 0])
    if axis == 0:
        return -math.atan2(-R[2, 1], R[1, 1])
    if axis == 1:
        return math.atan2(-R[0, 2], t)
    return 0.0


def compute_twist(R):
    return 2.0 * math.atan2(R[0, 2] - R[2, 0], R[0, 0] + R[1, 1] + R[2, 2] + 1)


class Segment:
    """IK_QRevoluteSegment (axis 0/1/2) or IK_QNullSegment (axis None)."""

    def __init__(self, axis, start, rest_basis, basis, length):
        self.axis, self.ndof = axis, (0 if axis is None else 1)
        self.start = np.asarray(start, float)
        self.rest = np.asarray(rest_basis, float)
        self.translation = np.array([0.0, length, 0.0])
        self.max_ext = float(np.linalg.norm(self.start)) + length
        self.orig_basis = np.asarray(basis, float).copy()
        self.limit, self.min, self.max, self.weight, self.locked = False, 0.0, 0.0, 1.0, False
        self.angle = self.new_angle = 0.0
        if axis is None:
            self.basis = self.orig_basis.copy()
        else:
            self.angle = compute_twist(basis) if axis == 1 else euler_angle_from_matrix(basis, axis)
            self.basis = rot_axis(self.angle, axis)
        self.dof_id = 0

    def set_limit(self, axis, lo, hi):
        if lo > hi or axis != self.axis:
            return
        self.min, self.max, self.limit = max(-math.pi, min(math.pi, lo)), max(-math.pi, min(math.pi, hi)), True

    def set_stiffness(self, axis, stiffness):
        if stiffness < 0.0 or axis != self.axis:
            return
        self.weight = 1.0 - min(stiffness, 1.0 - 0.001)

    def basis_change(self):
        return self.orig_basis.T @ self.basis


def _invert_sdls(J, beta, weight, wsqrt):
    task, dof = J.shape
    norm = np.zeros(dof)
    for j in range(0, task, 3):
        norm += np.sqrt((J[j:j + 3] ** 2).sum(0))
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    d_theta = np.zeros(dof)
    for i in range(len(S)):
        if S[i] <= 1e-10:
            continue
        w_inv = 1.0 / S[i]
        u = U[:, i]
        alpha = float(u @ beta) * w_inv
        N = sum(math.sqrt(float(u[j:j + 3] @ u[j:j + 3])) for j in range(0, task, 3))
        v = Vt[i]
        M = float(np.abs(v) @ norm) * w_inv
        tmp = v * alpha
        max_dtheta = float(np.max(np.abs(tmp) * wsqrt)) if dof else 0.0
        gamma = MAX_ANGLE_CHANGE * (N / M if N < M else 1.0)
        damp = gamma / max_dtheta if gamma < max_dtheta else 1.0
        dofdamp = np.minimum(damp / weight, 1.0)
        d_theta += 0.80 * dofdamp * tmp
    d_theta *= weight
    mx = float(np.max(np.abs(d_theta))) if dof else 0.0
    if mx > MAX_ANGLE_CHANGE:
        d_theta *= MAX_ANGLE_CHANGE / (MAX_ANGLE_CHANGE + mx)
    return d_theta


def solve(segs, pos_goal=None, pos_weight=0.0, rot_goal=None, rot_weight=0.0, iterations=500):
    """Solve the chain (segs root first, tip last) in place; returns the iterations used. IK_Solve / Setup / Solve."""
    tasks = []
    if pos_goal is not None and pos_weight != 0.0:
        tasks.append(["pos", np.asarray(pos_goal, float).copy(), pos_weight])
    if rot_goal is not None and rot_weight != 0.0:
        tasks.append(["rot", np.asarray(rot_goal, float), rot_weight])
    if not tasks:
        return 0
    dof = 0
    for s in segs:
        s.dof_id, dof = dof, dof + s.ndof
    if dof == 0:
        return 0
    tw = sum(t[2] for t in tasks)
    for t in tasks:
        t[2] /= tw                                            # task weights sum to 1
    clamp = sum(s.max_ext for s in segs) / (2 * len(segs))    # IK_QPositionTask clamp length
    total = sum(s.max_ext for s in segs)
    scale = 1.0 / total if total else 1.0
    for s in segs:
        s.start, s.translation = s.start * scale, s.translation * scale
    clamp *= scale
    for t in tasks:
        if t[0] == "pos":
            t[1] = t[1] * scale
    weight = np.ones(dof)
    for s in segs:
        if s.ndof:
            weight[s.dof_id] = s.weight
    wsqrt = np.sqrt(weight)
    task_size = 3 * len(tasks)
    used = 0
    for it in range(iterations):
        used = it + 1
        R, t0 = np.eye(3), np.zeros(3)                       # root matrix: identity
        for s in segs:                                       # UpdateTransform down the chain
            s.gstart = t0 + R @ s.start
            s.gR = R @ s.rest @ s.basis
            s.gend = s.gstart + s.gR @ s.translation
            R, t0 = s.gR, s.gend
        tip = segs[-1]
        J, beta = np.zeros((task_size, dof)), np.zeros(task_size)
        for k, (kind, goal, w) in enumerate(tasks):
            r = slice(3 * k, 3 * k + 3)
            if kind == "pos":
                d = goal - tip.gend
                ln = float(np.linalg.norm(d))
                if ln > clamp:
                    d = d * (clamp / ln)
                beta[r] = w * d
                for s in segs:
                    if s.ndof:
                        J[r, s.dof_id] = np.cross(s.gstart - tip.gend, s.gR[:, s.axis] * w) * wsqrt[s.dof_id]
            else:
                drm = (goal @ tip.gR.T).T
                beta[r] = w * -0.5 * np.array([drm[2, 1] - drm[1, 2], drm[0, 2] - drm[2, 0], drm[1, 0] - drm[0, 1]])
                for s in segs:
                    if s.ndof:
                        J[r, s.dof_id] = s.gR[:, s.axis] * w * wsqrt[s.dof_id]
        norm = 0.0
        while True:                                          # invert, clamp to limits, lock, repeat
            d_theta = _invert_sdls(J, beta, weight, wsqrt)
            minabs, minseg, mindelta, locked = 1e10, None, 0.0, False
            for s in segs:
                if not s.ndof or s.locked:
                    continue
                s.new_angle = s.angle + d_theta[s.dof_id]
                if not s.limit:
                    continue
                if s.new_angle > s.max:
                    delta = s.max - s.angle
                elif s.new_angle < s.min:
                    delta = s.min - s.angle
                else:
                    continue
                s.new_angle = s.angle + delta
                if abs(delta) < IK_EPSILON:
                    s.locked, locked = True, True
                    beta -= J[:, s.dof_id] * delta
                    J[:, s.dof_id] = 0.0
                elif abs(delta) < minabs:
                    minabs, minseg, mindelta = abs(delta), s, delta
            if minseg is not None:
                minseg.locked, locked = True, True
                beta -= J[:, minseg.dof_id] * mindelta
                J[:, minseg.dof_id] = 0.0
                norm = max(norm, minabs)
            if not locked:
                for s in segs:
                    s.locked = False
                    if s.ndof:
                        s.angle = s.new_angle
                        s.basis = rot_axis(s.angle, s.axis)
                break
        for s in segs:
            s.locked = False
        norm = max(norm, float(np.max(np.abs(d_theta))))
        if norm < 1e-3 and it > 10:
            break
    for s in segs:                                           # Scale(1 / scale)
        s.start, s.translation = s.start / scale, s.translation / scale
    return used
