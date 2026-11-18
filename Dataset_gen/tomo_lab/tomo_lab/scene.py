"""Stage helpers for the tomo packing scene under Isaac Lab (run inside a launched SimulationApp).

The scene USD was built for the physics-OFF renderer. These helpers make it usable with physics ON: the same
dataset-matching fixes (ported from runtime/isaac_runtime.py: apply_scene_fixes), the phone's box-centre frame
(same convention as the recordings) and a surface (suction) gripper on the robot's suction link, built like
NVIDIA's reference gripper (Isaac Lab test_gripper.usd: an IsaacSurfaceGripper prim + a D6 attachment joint).
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from .paths import rot_xyz


def gf_to_np(m) -> np.ndarray:
    return np.array(m, dtype=float).T


def np_to_gf(M) -> Gf.Matrix4d:
    return Gf.Matrix4d(np.asarray(M, dtype=float).T.tolist())


def world(prim) -> np.ndarray:
    if not prim or not prim.IsValid() or prim.IsPseudoRoot():
        return np.eye(4)
    return gf_to_np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def set_world(prim, W: np.ndarray) -> None:
    xf = UsdGeom.Xformable(prim)
    local = np.linalg.inv(world(prim.GetParent())) @ W
    xf.ClearXformOpOrder()
    xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(np_to_gf(local))


def bbox_world(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return None if r.IsEmpty() else (np.array(r.GetMin()), np.array(r.GetMax()))


def apply_scene_fixes(stage, root: str, fx: dict) -> list[str]:
    """Geometry part of the existing renderer's scene_fixes (translate_world, fit_bbox_xy, hide)."""
    done = []
    for path, d in (fx.get("translate_world") or {}).items():
        prim = stage.GetPrimAtPath(root + path)
        T = np.eye(4)
        T[:3, 3] = [float(v) for v in d]
        set_world(prim, T @ world(prim))
        done.append(f"moved {path} {list(map(float, d))}")
    for path, box in (fx.get("fit_bbox_xy") or {}).items():
        prim = stage.GetPrimAtPath(root + path)
        lo, hi = bbox_world(prim)
        A = np.eye(4)
        for ax, key in ((0, "x"), (1, "y")):
            n_lo, n_hi = (float(v) for v in box.get(key, (lo[ax], hi[ax])))
            A[ax, ax] = (n_hi - n_lo) / (hi[ax] - lo[ax])
            A[ax, 3] = n_lo - A[ax, ax] * lo[ax]
        set_world(prim, A @ world(prim))
        done.append(f"fit {path}")
    for path in fx.get("hide") or []:
        prim = stage.GetPrimAtPath(root + path)
        if prim.IsValid():
            prim.SetActive(False)                 # inactive: neither rendered nor simulated
            done.append(f"removed {path}")
    return done


def phone_object_frame(stage, mesh_path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Box-centre frame of the phone mesh (x = long axis) in world, and its size (w, l, h) - as the recordings."""
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
    pts = np.asarray(mesh.GetPointsAttr().Get(), float)
    W = (world(mesh.GetPrim()) @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    z_top, z_bot = W[:, 2].max(), W[:, 2].min()
    top = W[W[:, 2] > z_top - 0.002][:, :2]
    c = top.mean(0)
    _, V = np.linalg.eigh(np.cov((top - c).T))
    yaw = math.atan2(V[1, 1], V[0, 1])
    ext = (W[:, :2] - c) @ V
    F = np.eye(4)
    F[:3, :3] = rot_xyz(0, 0, yaw)
    F[:3, 3] = [c[0], c[1], 0.5 * (z_top + z_bot)]
    return F, (float(np.ptp(ext[:, 1])), float(np.ptp(ext[:, 0])), float(z_top - z_bot))


def quat_z_to(d: np.ndarray) -> Gf.Quatf:
    """Rotation taking +Z onto unit vector d (w, x, y, z)."""
    z = np.array([0.0, 0.0, 1.0])
    d = d / np.linalg.norm(d)
    axis = np.cross(z, d)
    s, c = np.linalg.norm(axis), float(z @ d)
    if s < 1e-9:
        return Gf.Quatf(1, 0, 0, 0) if c > 0 else Gf.Quatf(0, 1, 0, 0)
    axis /= s
    half = math.atan2(s, c) / 2
    return Gf.Quatf(math.cos(half), *(axis * math.sin(half)).tolist())


def add_suction_gripper(stage, link_path: str, cup_local: np.ndarray, cup_dir_local: np.ndarray,
                        max_grip_distance: float = 0.02, force_limit: float = 200.0, body1: str | None = None) -> str:
    """IsaacSurfaceGripper on `link_path` with one D6 attachment point at `cup_local` (link frame), gripping
    along `cup_dir_local`. Mirrors NVIDIA's test_gripper.usd. Returns the gripper prim path."""
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    for ext in ("isaacsim.robot.schema", "isaacsim.robot.surface_gripper"):   # not loaded by a plain Isaac Lab app
        mgr.set_extension_enabled_immediate(ext, True)
    from usd.schema.isaac import robot_schema

    joint_path = f"{link_path}/SuctionAttachment"
    j = UsdPhysics.Joint.Define(stage, joint_path)
    jp = j.GetPrim()
    for ax in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        lim = UsdPhysics.LimitAPI.Apply(jp, ax)          # low > high = locked axis, as in the reference
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)
    lim = UsdPhysics.LimitAPI.Get(jp, "transZ")
    lim.GetLowAttr().Set(0.0)
    lim.GetHighAttr().Set(-0.005)                        # a little give along the cup axis
    robot_schema.ApplyAttachmentPointAPI(jp)
    jp.CreateAttribute("isaac:forwardAxis", Sdf.ValueTypeNames.Token).Set("Z")
    jp.CreateAttribute("isaac:clearanceOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    j.CreateBody0Rel().SetTargets([link_path])
    if body1:                                            # as in the reference: body1 = another robot link until a grip
        j.CreateBody1Rel().SetTargets([body1])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*np.asarray(cup_local, float).tolist()))
    j.CreateLocalRot0Attr().Set(quat_z_to(np.asarray(cup_dir_local, float)))
    j.CreateExcludeFromArticulationAttr().Set(True)
    j.CreateBreakForceAttr().Set(3.4028235e38)
    j.CreateBreakTorqueAttr().Set(3.4028235e38)

    grip_path = f"{link_path}/SurfaceGripper"
    g = robot_schema.CreateSurfaceGripper(stage, grip_path)
    g.GetRelationship("isaac:attachmentPoints").SetTargets([joint_path])
    g.GetAttribute("isaac:maxGripDistance").Set(max_grip_distance)
    g.GetAttribute("isaac:coaxialForceLimit").Set(force_limit)
    g.GetAttribute("isaac:shearForceLimit").Set(force_limit)
    g.GetAttribute("isaac:retryInterval").Set(0.2)
    return grip_path


def move_articulation_root(stage, from_path: str, to_path: str) -> None:
    """The scene's ArticulationRootAPI sits on a Scope (/tomo_re_uhand/Geometry), which is not Xformable, so PhysX
    builds no articulation and none of the robot's links become bodies ('CreateJoint - no bodies defined'). It was
    never noticed because the old renderer ran with physics off. Put it on the robot's Xform root instead (the
    parent of both the links and the fixed root_joint)."""
    src, dst = stage.GetPrimAtPath(from_path), stage.GetPrimAtPath(to_path)
    if src.HasAPI(UsdPhysics.ArticulationRootAPI):
        src.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(dst)


def box_collider_for(stage, prim_path: str, collider_path: str, thickness: float = 0.04) -> tuple:
    """Turn off a prim's own collision and add a static box with the same XY extent whose top is the prim's top
    (the table mesh's collider is invalid once scene_fixes stretches it). Returns (min, max) of the new box."""
    prim = stage.GetPrimAtPath(prim_path)
    for p in Usd.PrimRange(prim):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    lo, hi = bbox_world(prim)
    lo = np.array([lo[0], lo[1], hi[2] - thickness])
    cube = UsdGeom.Cube.Define(stage, collider_path)
    cube.CreateSizeAttr(1.0)
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)               # not rendered
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*((lo + hi) / 2).tolist()))
    xf.AddScaleOp().Set(Gf.Vec3f(*(hi - lo).tolist()))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return lo, hi


def load_scene_sublayer(stage, usd_path: str) -> None:
    """Bring in the whole scene file as a sublayer: every root prim keeps its path (/tomo_re_uhand, /Iphone, /Env,
    ...) and its relative asset paths anchor to the file, as when the old runtime opened it directly. (A reference
    would bring only the file's default prim.)"""
    stage.GetRootLayer().subLayerPaths.append(str(usd_path))


def deactivate_extra_physics_scenes(stage, keep: str = "/physicsScene") -> list[str]:
    """The scene file may carry its own PhysicsScene; Isaac Lab owns /physicsScene."""
    off = []
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene) and str(p.GetPath()) != keep:
            p.SetActive(False)
            off.append(str(p.GetPath()))
    return off
