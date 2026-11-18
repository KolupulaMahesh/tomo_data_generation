"""Render one episode in Isaac Sim (2026-10-29, the generic renderer): the scene's look (Blender's own USD export,
blender_scripts/export_scene_usd.py) + the motion our rig computed (make_motion.py: every moving object, every bone of
the other armatures, the robot's 46 joints), written as time samples into ONE file, <episode>/episode.usdc - open it in
Isaac Sim and play the timeline to watch the episode - then every frame rendered from the robot's head camera.
Nothing here is about one scene: what the .blend cannot tell comes from scenes/<scene>.yaml via <episode>/episode.json.
The robot, its kinematic drive, its head camera and the light calibration are the dataset generator's
(renderer/tools/fs_isaac_gen.py), unchanged; the robot's own USD replaces the Blender robot.

    ~/isaacsim/python.sh render_episode.py <episode dir> [--frames 0,300] [--stride 1] [--W 960] [--warm 2] [--build-only]
-> <episode dir>/episode.usdc + build_check.json (the file's poses vs the rig's); raw/rgb_XXXX.png + raw/depth_XXXX.npy
   (mm at the packed size, 0 = no reading); labels_raw.npz (robot joints, every moving object's and the camera's world
   matrix per frame); variant.json
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("episode")
ap.add_argument("--frames", default="", help="only these frames (comma list)")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--W", type=int, default=960, help="render width (height from the camera's 16:9)")
ap.add_argument("--warm", type=int, default=2, help="renderer updates per frame before the capture")
ap.add_argument("--build-only", action="store_true", help="write episode.usdc + its check, no render")
ap.add_argument("--usd-out", default="", help="write the playable file here instead of <episode>/episode.usdc "
                "(e.g. next to the scene's scene.usdc, user 2026-11-18)")
ap.add_argument("--last-frame", type=int, default=-1, help="render only up to this frame (e.g. the end of a step)")
a = ap.parse_args()

# ---------------- settings: the robot (as the dataset generator) ----------------
HERE = Path(__file__).resolve().parent
EP = Path(a.episode).resolve()
USD_DIR = Path(a.usd_out).resolve().parent if a.usd_out else EP     # relative paths in the file are from where it lies
ROBOT_USD = HERE / "renderer/robot_scene/tomo_packing_suction.usd"
ROBOT_PRIM, ROBOT_ROOT = "/tomo_re_uhand", "/tomo_re_uhand/Geometry"
CAM = ROBOT_ROOT + "/Body_base/Head_link1/Head_link2/Head_camera_link/robot_view_cam"
ROBOT_EXTRA = ["/tomo_re_uhand/Box_1_v1___open"]                    # a carton inside the robot's file: not drawn
K = (1070.789795, 1070.526978, 962.309387, 550.0, 1920, 1080)       # the head camera: fx fy cx cy W H
PACK = (960, 540)                                                   # saved size (the live camera pipeline's)
CHECK_FRAMES = 12                                                   # frames compared in build_check.json

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
import cv2  # noqa: E402
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdSkel, Vt  # noqa: E402

CFG = json.loads((EP / "episode.json").read_text())
LOOK_DIR = (EP / CFG["look_dir"]).resolve()
LOOK = json.loads((LOOK_DIR / "export.json").read_text())
MO = np.load(EP / CFG["motion"])
FR = [int(x) for x in MO["frames"]]
OBJ = LOOK["objects"]
ROBOT_B = OBJ[CFG["robot"]]                                          # the Blender robot's prim (replaced, unless drawn "scene")
ROBOT_DRAWN = CFG.get("robot_drawn", "usd")                          # "scene": the .blend's own robot, moved like every other object
if CFG.get("camera"):                                                # the scene file's camera and the real camera's lens
    K, CAM = tuple(CFG["camera"]["lens"]), OBJ[CFG["camera"]["object"]]
    PACK = (PACK[0], int(round(PACK[0] * K[5] / K[4])))             # saved size keeps the camera's shape (4:3 -> 960x720)
if LOOK["geometry_changes"]:
    raise SystemExit(f"meshes change shape over time other than by bones - the static look cannot show that: "
                     f"{LOOK['geometry_changes'][:5]}")


def m4(M):
    return Gf.Matrix4d(*np.asarray(M, float).T.flatten().tolist())     # numpy column vectors -> Gf row vectors


def np4(G):
    return np.array(G, float).T


def inside(path, root):
    return path == root or path.startswith(root + "/")


def replaced(p):
    """inside the Blender robot, when the robot's own USD replaces it."""
    return ROBOT_DRAWN == "usd" and inside(p, ROBOT_B)


def rel(p):
    return os.path.relpath(p, USD_DIR)


# ---------------- the robot (fs_isaac_gen.py's kinematic scheme: translate lp0 + orient lr0*R(axis,q) per link) -------
AX = {"X": np.array([1., 0, 0]), "Y": np.array([0, 1., 0]), "Z": np.array([0, 0, 1.])}


def robot_links(st):
    J = {}
    for prim in st.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint)):
            continue
        j = UsdPhysics.Joint(prim)
        b0 = [str(x) for x in j.GetBody0Rel().GetTargets()]
        b1 = [str(x) for x in j.GetBody1Rel().GetTargets()]
        if not b0 or not b1 or not b1[0].startswith(ROBOT_ROOT):
            continue
        child = st.GetPrimAtPath(b1[0])
        if not child.IsValid() or child.GetParent().GetPath().pathString != b0[0]:
            continue
        rev = prim.IsA(UsdPhysics.RevoluteJoint)
        lr0 = j.GetLocalRot0Attr().Get()
        mr = prim.GetRelationship("newton:mimicJoint")
        mim = (mr.GetTargets()[0].name, float(prim.GetAttribute("newton:mimicCoef0").Get() or 0),
               float(prim.GetAttribute("newton:mimicCoef1").Get() or 1)) if mr and mr.GetTargets() else None
        xf = UsdGeom.Xformable(child)
        scale = np.ones(3)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale and op.Get() is not None:
                scale = np.array([float(v) for v in op.Get()])
        t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "kin")
        o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "kin")
        ops = [t_op, o_op]
        if np.abs(scale - 1).max() > 1e-6:
            s_op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "kin")
            s_op.Set(Gf.Vec3d(*scale))
            ops.append(s_op)
        xf.SetXformOpOrder(ops, xf.GetResetXformStack())
        t_op.Set(Gf.Vec3d(*[float(x) for x in j.GetLocalPos0Attr().Get()]))
        J[prim.GetName()] = dict(rev=rev, axis=AX[str(prim.GetAttribute("physics:axis").Get() or "Z")] if rev else AX["Z"],
                                 lr0=(float(lr0.GetReal()), [float(x) for x in lr0.GetImaginary()]), mim=mim, o_op=o_op)
    return J


def jval(J, n, qd):
    d = J[n]
    if not d["rev"]:
        return 0.0
    if n in qd:
        return qd[n]
    return d["mim"][1] + d["mim"][2] * qd.get(d["mim"][0], 0.0) if d["mim"] else 0.0


def physics_off(st):
    """Purely kinematic: no PhysX anywhere (else the referenced articulation gets simulated and drags the robot)."""
    for p in list(st.Traverse()):
        if p.IsA(UsdPhysics.Scene):
            p.SetActive(False)
            continue
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
    if st.GetPrimAtPath(ROBOT_PRIM + "/Physics").IsValid():
        st.GetPrimAtPath(ROBOT_PRIM + "/Physics").SetActive(False)


# ---------------- the look the exporter cannot carry, and this episode's look ----------------
def self_lit(st, e):
    """Cycles shows an image wired into the surface self-lit: the same as a preview surface with the image as diffuse
    and emissive colour (the dataset generator's room-backdrop material)."""
    base = e["prim"] + "/self_lit"
    sh = UsdShade.Shader.Define(st, base + "/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    tx = UsdShade.Shader.Define(st, base + "/Tex")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel(LOOK_DIR / e["image"])))
    if e["strength"] != 1.0:
        tx.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*[e["strength"]] * 3, 1.0))
    rd = UsdShade.Shader.Define(st, base + "/UV")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    rgb = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    UsdShade.Material(st.GetPrimAtPath(e["prim"])).CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")


def meshes_of(st, name):
    return [p for p in Usd.PrimRange(st.GetPrimAtPath(OBJ[name])) if p.IsA(UsdGeom.Mesh)]


def recolor(st, name, rgb):
    mat = UsdShade.Material.Define(st, f"/FS_Looks/{Tf_ident(name)}")
    sh = UsdShade.Shader.Define(st, f"/FS_Looks/{Tf_ident(name)}/S")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    for p in meshes_of(st, name):
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants)


def retexture(st, name, image, hue):
    """The object's base-colour textures (the image feeding each of its materials' diffuseColor) -> image, hue-shifted."""
    shaders = set()
    for p in meshes_of(st, name):
        for q in [p] + [c for c in p.GetChildren() if c.IsA(UsdGeom.Subset)]:
            m, _ = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial()
            if not m:
                continue
            for s in Usd.PrimRange(m.GetPrim()):
                sh = UsdShade.Shader(s)
                if not s.IsA(UsdShade.Shader) or sh.GetIdAttr().Get() != "UsdPreviewSurface":
                    continue
                dc = sh.GetInput("diffuseColor")
                src = dc.GetConnectedSource() if dc else None
                if src and UsdShade.Shader(src[0].GetPrim()).GetIdAttr().Get() == "UsdUVTexture":
                    shaders.add(str(src[0].GetPrim().GetPath()))
    if image:
        im = cv2.imread(str((EP / image).resolve()), cv2.IMREAD_COLOR)
    else:                                                           # no image: the object's own print
        fa = UsdShade.Shader(st.GetPrimAtPath(sorted(shaders)[0])).GetInput("file").Get()
        im = cv2.imread(fa.resolvedPath or fa.path, cv2.IMREAD_COLOR)
    if hue:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = ((hsv[..., 0].astype(int) + int(hue / 2)) % 180).astype(np.uint8)
        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    dst = EP / f"print_{Tf_ident(name)}.png"
    cv2.imwrite(str(dst), im)
    for s in sorted(shaders):
        UsdShade.Shader(st.GetPrimAtPath(s)).GetInput("file").Set(Sdf.AssetPath(rel(dst)))
    return sorted(shaders)


def Tf_ident(name):
    """A Blender name as a valid USD prim / file name part ("Box 1_v1.001" -> "Box_1_v1_001")."""
    return Tf.MakeValidIdentifier(name)


# ---------------- the episode file ----------------
def add_robot_usd(st, root, T):
    """The robot's own USD in the Blender robot's place, its joints written as time samples, no physics anywhere."""
    rp = st.DefinePrim(ROBOT_PRIM, "Xform")
    rp.GetReferences().AddReference(rel(ROBOT_USD), ROBOT_PRIM)
    for x in ROBOT_EXTRA:
        if st.GetPrimAtPath(x).IsValid():
            st.GetPrimAtPath(x).SetActive(False)
    RW = MO["robot_world"]
    if np.abs(RW - np.eye(4)).max() > 1e-9:                             # the Blender robot stands elsewhere
        xf = UsdGeom.Xformable(rp)
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "scene")
        xf.SetXformOpOrder([op] + [o for o in xf.GetOrderedXformOps() if o.GetName() != op.GetName()])
        for i, f in enumerate(FR):
            op.Set(m4(RW[i]), T(f))
    J = robot_links(st)
    names = [str(n) for n in MO["robot_joints"]]
    assert all(n in J and J[n]["rev"] for n in names), "robot joints missing"
    qds = [dict(zip(names, qi)) for qi in MO["robot_q"]]
    with Sdf.ChangeBlock():
        for n, d in J.items():
            lr = Gf.Quatd(d["lr0"][0], Gf.Vec3d(*d["lr0"][1]))
            if not d["rev"]:
                d["o_op"].Set(lr)
                continue
            attr = d["o_op"].GetAttr().GetPath()
            for f, qd in zip(FR, qds):
                rot = Gf.Rotation(Gf.Vec3d(*d["axis"]), math.degrees(jval(J, n, qd))).GetQuat()
                root.SetTimeSample(attr, float(f), lr * rot)
    physics_off(st)


def camera_prim(st, path):
    """The camera itself: Blender's exporter writes an object's transform and its camera data as two prims, one inside the other."""
    return next(str(p.GetPath()) for p in Usd.PrimRange(st.GetPrimAtPath(path)) if p.IsA(UsdGeom.Camera))


def build(path):
    global CAM
    if path.exists():
        path.unlink()
    st = Usd.Stage.CreateNew(str(path))
    root = st.GetRootLayer()
    root.subLayerPaths.append(rel(LOOK_DIR / LOOK["usd"]))
    fps = float(MO["fps"])
    st.SetStartTimeCode(FR[0]); st.SetEndTimeCode(FR[-1])
    st.SetTimeCodesPerSecond(fps); st.SetFramesPerSecond(fps)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(st, 1.0)
    root.defaultPrim = "root"
    T = lambda f: Usd.TimeCode(float(f))                                # noqa: E731
    xc0 = UsdGeom.XformCache(Usd.TimeCode.Default())                    # the static scene, before any change
    static_world = lambda p: np4(xc0.GetLocalToWorldTransform(p))       # noqa: E731

    # the objects the scene file says to leave out; Blender robot out and the robot's own USD in its place (unless drawn "scene")
    if ROBOT_DRAWN == "usd":
        st.GetPrimAtPath(ROBOT_B).SetActive(False)
    for n in CFG["hide"]:
        st.GetPrimAtPath(OBJ[n]).SetActive(False)
    if ROBOT_DRAWN == "usd":
        add_robot_usd(st, root, T)

    # every moving object: its local matrix under its USD parent, from the rig's world matrices
    on = [str(n) for n in MO["object_names"]]
    W = MO["object_world"]
    by_prim = {OBJ[n]: k for k, n in enumerate(on)}
    written = []
    for k, n in sorted(enumerate(on), key=lambda x: OBJ[x[1]].count("/")):
        prim = st.GetPrimAtPath(OBJ[n])
        par = prim.GetParent()
        pp = str(par.GetPath())
        if pp in by_prim:
            Wp = W[:, by_prim[pp]]
        elif any(inside(pp, x) for x in by_prim):
            raise NotImplementedError(f"{n}: its USD parent {pp} moves but is not a Blender object")
        else:
            Wp = np.broadcast_to(static_world(par), W[:, k].shape)
        L = np.linalg.inv(Wp) @ W[:, k]
        L0 = np4(UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default()))
        if np.abs(L - L[:1]).max() < 1e-9 and np.abs(L[0] - L0).max() < 1e-6:
            continue                                                    # rides on its parent: the export is right
        xf = UsdGeom.Xformable(prim)
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "motion")
        xf.SetXformOpOrder([op], xf.GetResetXformStack())
        attr = op.GetAttr().GetPath()
        with Sdf.ChangeBlock():
            for i, f in enumerate(FR):
                root.SetTimeSample(attr, float(f), m4(L[i]))
        written.append(n)

    # every skeleton outside the robot: joint local transforms (parent pose^-1 @ pose, Blender's exporter's convention)
    bidx = {(str(o), str(b)): i for i, (o, b) in enumerate(zip(MO["bone_owner"], MO["bone_names"]))}
    skels = []
    for s in LOOK["skeletons"]:
        if replaced(s["prim"]):
            continue
        joints = s["joints"]
        par = [joints.index(j.rsplit("/", 1)[0]) if "/" in j else -1 for j in joints]
        P = MO["bone_pose"][:, [bidx[(s["armature"], b)] for b in s["bones"]]]
        L = P.copy()
        for j, pj in enumerate(par):
            if pj >= 0:
                L[:, j] = np.linalg.inv(P[:, pj]) @ P[:, j]
        anim = UsdSkel.Animation.Define(st, s["prim"] + "/motion")
        anim.CreateJointsAttr(Vt.TokenArray(joints))
        attrs = (anim.CreateTranslationsAttr(), anim.CreateRotationsAttr(), anim.CreateScalesAttr())
        moves = np.abs(L - L[:1]).max() > 0
        for i, f in enumerate(FR if moves else FR[:1]):
            vals = UsdSkel.DecomposeTransforms(Vt.Matrix4dArray([m4(x) for x in L[i]]))
            for at, v in zip(attrs, vals):
                at.Set(v, T(f) if moves else Usd.TimeCode.Default())
        UsdSkel.BindingAPI.Apply(st.GetPrimAtPath(s["prim"])).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
        skels.append(s["armature"])

    # objects not drawn from a frame on
    for n, f in CFG["hide_from_frame"].items():
        vis = UsdGeom.Imageable(st.GetPrimAtPath(OBJ[n])).GetVisibilityAttr()
        vis.Set(UsdGeom.Tokens.inherited, T(FR[0]))
        vis.Set(UsdGeom.Tokens.invisible, T(f))

    # look: light calibration, the self-lit images, this episode's colours and prints
    for p in st.Traverse():
        ia = p.GetAttribute("inputs:intensity")
        if p.HasAPI(UsdLux.LightAPI) and ia and ia.Get() is not None and not inside(str(p.GetPath()), ROBOT_PRIM):
            ia.Set(float(ia.Get()) * CFG["light_gain"])
    if CFG["dome"]:
        UsdLux.DomeLight.Define(st, "/FS_Dome").CreateIntensityAttr(CFG["dome"])
    for e in LOOK["emissive"]:
        self_lit(st, e)
    for n, rgb in CFG["recolor"].items():
        recolor(st, n, rgb)
    prints = {n: retexture(st, n, v["image"], v.get("hue", 0)) for n, v in CFG["retexture"].items()}

    # the head camera: the robot camera's intrinsics
    fx, fy, cx, cy, FW, FH = K
    CAM = camera_prim(st, CAM)
    cam = UsdGeom.Camera(st.GetPrimAtPath(CAM))
    hap = float(cam.GetHorizontalApertureAttr().Get() or 20.955)
    foc = hap * fx / FW
    vap = foc * FH / fy
    cam.GetFocalLengthAttr().Set(foc); cam.GetHorizontalApertureAttr().Set(hap); cam.GetVerticalApertureAttr().Set(vap)
    cam.GetHorizontalApertureOffsetAttr().Set((FW / 2 - cx) * hap / FW)
    cam.GetVerticalApertureOffsetAttr().Set((cy - FH / 2) * vap / FH)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    root.Save()

    # check: the saved file's poses vs the rig's (objects: world matrix; bones: armature-space pose)
    st = Usd.Stage.Open(str(path))
    cf = np.linspace(0, len(FR) - 1, CHECK_FRAMES).round().astype(int)
    obj_err, bone_err = 0.0, 0.0
    cache = UsdSkel.Cache()
    for i in cf:
        xc = UsdGeom.XformCache(T(FR[i]))
        for k, n in enumerate(on):
            obj_err = max(obj_err, float(np.abs(np4(xc.GetLocalToWorldTransform(st.GetPrimAtPath(OBJ[n]))) - W[i, k]).max()))
        for s in LOOK["skeletons"]:
            if replaced(s["prim"]):
                continue
            sq = cache.GetSkelQuery(UsdSkel.Skeleton(st.GetPrimAtPath(s["prim"])))
            X = np.array([np4(m) for m in sq.ComputeJointSkelTransforms(T(FR[i]))])
            P = MO["bone_pose"][i, [bidx[(s["armature"], b)] for b in s["bones"]]]
            bone_err = max(bone_err, float(np.abs(X - P).max()))
    chk = {"objects_written": written, "skeletons_written": skels, "prints": prints,
           "max_object_world_diff": obj_err, "max_bone_pose_diff": bone_err, "frames_checked": [FR[i] for i in cf]}
    (EP / "build_check.json").write_text(json.dumps(chk, indent=1))
    print("[render] episode file", path, json.dumps({k: v for k, v in chk.items() if k != "frames_checked"}), flush=True)


t0 = time.time()
USD_PATH = Path(a.usd_out).resolve() if a.usd_out else EP / "episode.usdc"
build(USD_PATH)
print(f"[render] built in {time.time() - t0:.1f} s", flush=True)
if a.build_only:
    app.close()
    raise SystemExit(0)

# ---------------- render ----------------
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(str(USD_PATH))
for _ in range(30):
    app.update()
st = ctx.get_stage()
W_, H_ = a.W, int(round(a.W * K[5] / K[4]))
rpd = rep.create.render_product(CAM, (W_, H_))
a_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
a_rgb.attach([rpd]); a_dep.attach([rpd])
tl = omni.timeline.get_timeline_interface()
tps = st.GetTimeCodesPerSecond() or 30.0
PW, PH = PACK
RAW = EP / "raw"
RAW.mkdir(exist_ok=True)
flist = [int(x) for x in a.frames.split(",")] if a.frames else \
    [f for f in FR[::a.stride] if a.last_frame < 0 or f <= a.last_frame]
camw = np.zeros((len(FR), 4, 4))
t1 = time.time()
for i, f in enumerate(flist):
    tl.set_current_time(f / tps)
    for _ in range(a.warm):
        app.update()
    rep.orchestrator.step(rt_subframes=4, pause_timeline=True, delta_time=0.0)    # 0: capture AT the set time
    # (None plays the timeline on before the capture: the picture lags the frame, and the last frame loops to 0)
    r = np.asarray(a_rgb.get_data())
    d = np.asarray(a_dep.get_data())
    camw[FR.index(f)] = np4(UsdGeom.XformCache(Usd.TimeCode(f)).GetLocalToWorldTransform(st.GetPrimAtPath(CAM)))
    cv2.imwrite(str(RAW / f"rgb_{f:04d}.png"), cv2.cvtColor(r[..., :3], cv2.COLOR_RGB2BGR))
    dm = np.clip(np.rint(np.where(np.isfinite(d) & (d > 0) & (d < 60.0), d, 0.0) * 1000), 0, 65535).astype(np.uint16)
    np.save(RAW / f"depth_{f:04d}.npy", cv2.resize(dm, (PW, PH), interpolation=cv2.INTER_NEAREST))
    if i % 100 == 0:
        print(f"[render] frame {f}/{FR[-1]}  {(time.time() - t1) / (i + 1):.2f} s/frame", flush=True)
tr = (CFG.get("offsets") or {}).get(CFG.get("track") or "")
off = CFG["phone_offset_dx_dy_dyaw"] or ([tr[0][0], tr[0][1], tr[1][2]] if tr else [0.0, 0.0, 0.0])
info = {"dx": off[0], "dy": off[1], "dyaw": off[2], "cam": "robot", "K_full": list(K), "pack_wh": list(PACK),
        "render_s": round(time.time() - t1, 1), "frames_rendered": len(flist), "last_frame": flist[-1],
        "renderer": "isaac-generic", "scene": CFG["scene"], "seed": CFG.get("seed"), "offsets": CFG.get("offsets"),
        "recolor": CFG["recolor"], "retexture": CFG["retexture"]}
np.savez_compressed(EP / "labels_raw.npz", names=MO["robot_joints"], q=MO["robot_q"].astype(np.float32),
                    frames=MO["frames"], cam_world=camw, object_names=MO["object_names"],
                    object_world=MO["object_world"], relay=MO["relay"],
                    **{k: MO[k] for k in MO.files if k.startswith("label_")})
(EP / "variant.json").write_text(json.dumps(info, indent=2))
print("[render] done", json.dumps(info), flush=True)
app.close()
