"""Full-sequence episode, ISAAC ONLY (no Blender): base scene + recorded motion exported once (isaac_base/), the TOMO robot
USD driven kinematically, right arm re-solved by IK for a new iPhone start pose, the phone carried by the new hand.
  ~/isaac-sim/python.sh fs_isaac_gen.py --out <episode_dir> --dx 0.03 --dy -0.02 --dyaw 10 [--stride 2]
      [--table-rgb r,g,b] [--phone-tex img] [--phone-hue deg]
then: ~/miniconda3/envs/tomo_qsa/bin/python fs_pack.py <episode_dir>"""
import argparse, os, json, math, time
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--dx", type=float, default=0.0); ap.add_argument("--dy", type=float, default=0.0); ap.add_argument("--dyaw", type=float, default=0.0)
ap.add_argument("--stride", type=int, default=2); ap.add_argument("--W", type=int, default=0, help="render width (0 = 1920 for the robot camera, 672 for original)"); ap.add_argument("--warm", type=int, default=8)
ap.add_argument("--light-gain", type=float, default=300.0); ap.add_argument("--dome", type=float, default=2500.0)
ap.add_argument("--table-rgb", default=""); ap.add_argument("--phone-tex", default=""); ap.add_argument("--phone-hue", type=float, default=0.0)
ap.add_argument("--hide", default="Plane_002,Plane_009", help="/root children to deactivate (the two 14 m animated planes fog the TOMO head camera)")
ap.add_argument("--ik-only", action="store_true", help="solve + check the arm for this phone pose, write variant.json, no render")
ap.add_argument("--cam", default="robot", choices=["original", "robot"], help="original = exact view of full_sequnece_001.mp4 (4:3); robot = user intrinsics 16:9")
ap.add_argument("--base", default="base.usdc", help="scene file in isaac_base/")
ap.add_argument("--frames", default="", help="debug: only these frames, saved as PNGs in <out>")
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # package root (finalfullsequncepraveen/)
BASE = os.path.join(ROOT, "isaac_base")
ROBOT_USD = os.path.join(ROOT, "robot_scene", "tomo_packing_suction.usd")
ROBOT_ROOT, CAM = "/tomo_re_uhand/Geometry", "/tomo_re_uhand/Geometry/Body_base/Head_link1/Head_link2/Head_camera_link/robot_view_cam"
HAND = "Right_hand"; R_ARM = list(range(9, 16))
GRASP_F, LIFT_F, REACH0, REACH1, CARRY1, RELEASE_F = 155, 160, 45, 132, 261, 290
K_ROBOT = (1070.789795, 1070.526978, 962.309387, 550.0, 1920, 1080)   # user's head camera (cy 550), 16:9
# the recording's Blender Head_Camera (lens 4.21 mm, sensor 7.504 mm, shift -0.004 / 0.012, 960x720) scaled to 1920 wide, 4:3
_f = 4.210000038146973 / 7.504000186920166 * 1920
K_ORIG = (_f, _f, 960 - (-0.004000000189989805) * 1920, 720 + 0.012000000104308128 * 1920, 1920, 1440)

K = K_ORIG if a.cam == "original" else K_ROBOT
PACK = (672, 504) if a.cam == "original" else (960, 540)   # robot: whole 1920x1080 frame saved at 960x540 (matches the live Tomo camera pipeline)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
import numpy as np, cv2, omni.usd, omni.timeline, omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdLux, UsdShade, UsdPhysics, Sdf, Gf
os.makedirs(a.out, exist_ok=True); RAW = os.path.join(a.out, "raw")
M0 = np.load(f"{BASE}/base_motion.npz")
NAMES = [str(x) for x in M0["names"]]; Q0 = M0["q"].astype(np.float64); P0 = M0["phone_world"]; T = len(Q0)

ctx = omni.usd.get_context(); ctx.open_stage(f"{BASE}/{a.base}")
for _ in range(30): app.update()
st = ctx.get_stage(); st.SetEditTarget(Usd.EditTarget(st.GetSessionLayer()))
st.GetPrimAtPath("/root/tomo").SetActive(False)
for _h in filter(None, a.hide.split(",")):
    if st.GetPrimAtPath("/root/" + _h).IsValid(): st.GetPrimAtPath("/root/" + _h).SetActive(False)                       # the baked Blender robot -> replaced by the TOMO USD
rp = st.DefinePrim("/tomo_re_uhand", "Xform"); rp.GetReferences().AddReference(ROBOT_USD, "/tomo_re_uhand")
for _ in range(10): app.update()
if st.GetPrimAtPath("/tomo_re_uhand/Box_1_v1___open").IsValid(): st.GetPrimAtPath("/tomo_re_uhand/Box_1_v1___open").SetActive(False)

# ---------------- kinematic robot (same scheme as runtime/isaac_runtime.py: translate lp0 + orient lr0*R(axis,q) per link) ----------------
AX = {"X": np.array([1., 0, 0]), "Y": np.array([0, 1., 0]), "Z": np.array([0, 0, 1.])}
def qmat(w, xyz):
    x, y, z = xyz; return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def rodr(ax, ang):
    K_ = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); return np.eye(3) + math.sin(ang) * K_ + (1 - math.cos(ang)) * K_ @ K_
J = {}
for prim in st.Traverse():
    if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint)): continue
    j = UsdPhysics.Joint(prim); b0 = [str(x) for x in j.GetBody0Rel().GetTargets()]; b1 = [str(x) for x in j.GetBody1Rel().GetTargets()]
    if not b0 or not b1 or not b1[0].startswith(ROBOT_ROOT): continue
    child = st.GetPrimAtPath(b1[0])
    if not child.IsValid() or child.GetParent().GetPath().pathString != b0[0]: continue
    rev = prim.IsA(UsdPhysics.RevoluteJoint); lr0 = j.GetLocalRot0Attr().Get()
    mr = prim.GetRelationship("newton:mimicJoint")
    mim = (mr.GetTargets()[0].name, float(prim.GetAttribute("newton:mimicCoef0").Get() or 0), float(prim.GetAttribute("newton:mimicCoef1").Get() or 1)) if mr and mr.GetTargets() else None
    lo, hi = (prim.GetAttribute("physics:lowerLimit").Get(), prim.GetAttribute("physics:upperLimit").Get()) if rev else (None, None)
    xf = UsdGeom.Xformable(child); scale = np.ones(3)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale and op.Get() is not None: scale = np.array([float(v) for v in op.Get()])
    t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "kin"); o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "kin"); ops = [t_op, o_op]
    if np.abs(scale - 1).max() > 1e-6:
        s_op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "kin"); s_op.Set(Gf.Vec3d(*scale)); ops.append(s_op)
    xf.SetXformOpOrder(ops, xf.GetResetXformStack())
    lp0 = np.array([float(x) for x in j.GetLocalPos0Attr().Get()]); t_op.Set(Gf.Vec3d(*lp0))
    J[prim.GetName()] = dict(child=child, path=child.GetPath().pathString, parent=b0[0], rev=rev, axis=AX[str(prim.GetAttribute("physics:axis").Get() or "Z")] if rev else AX["Z"],
                             lp0=lp0, lr0=(float(lr0.GetReal()), [float(x) for x in lr0.GetImaginary()]), mim=mim, o_op=o_op, scale=scale,
                             lim=(math.radians(lo), math.radians(hi)) if rev and lo is not None and abs(lo) < 1e6 else None)
assert all(n in J and J[n]["rev"] for n in NAMES), "robot joints missing"
# purely kinematic replay: no PhysX anywhere (otherwise the referenced articulation gets simulated and drags the robot away)
n_off = 0
for p in list(st.Traverse()):
    if p.IsA(UsdPhysics.Scene): p.SetActive(False); n_off += 1; continue
    if p.HasAPI(UsdPhysics.RigidBodyAPI): UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False); n_off += 1
    if p.HasAPI(UsdPhysics.ArticulationRootAPI): p.RemoveAPI(UsdPhysics.ArticulationRootAPI); n_off += 1
    if p.HasAPI(UsdPhysics.CollisionAPI): UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
if st.GetPrimAtPath("/tomo_re_uhand/Physics").IsValid(): st.GetPrimAtPath("/tomo_re_uhand/Physics").SetActive(False)
print(f"[gen] physics off ({n_off} prims)", flush=True)
LIM = np.array([J[n]["lim"] if J[n]["lim"] else (-1e9, 1e9) for n in NAMES])
by_child = {d["path"]: n for n, d in J.items()}
def chain(link_name):
    p = next(d["path"] for d in J.values() if d["path"].endswith("/" + link_name)); out = []
    while p in by_child: out.append(by_child[p]); p = J[by_child[p]]["parent"]
    return list(reversed(out)), p
CH, ROOT_LINK = chain(HAND)
ROOT_W = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(st.GetPrimAtPath(ROOT_LINK))).T
def jval(n, qd):
    d = J[n]
    if not d["rev"]: return 0.0
    if n in qd: return qd[n]
    return d["mim"][1] + d["mim"][2] * qd.get(d["mim"][0], 0.0) if d["mim"] else 0.0
def fk(q):
    qd = dict(zip(NAMES, q)); M = ROOT_W.copy()
    for n in CH:
        d = J[n]; L = np.eye(4); L[:3, 3] = d["lp0"]; L[:3, :3] = qmat(d["lr0"][0], d["lr0"][1]) @ rodr(d["axis"], jval(n, qd)) @ np.diag(d["scale"]); M = M @ L
    M[:3, :3] /= np.linalg.norm(M[:3, :3], axis=0); return M
def drive(q):
    qd = dict(zip(NAMES, q))
    with Sdf.ChangeBlock():
        for n, d in J.items():
            r = Gf.Quatd(d["lr0"][0], Gf.Vec3d(*d["lr0"][1]))
            if d["rev"]: r = r * Gf.Rotation(Gf.Vec3d(*d["axis"]), math.degrees(jval(n, qd))).GetQuat()
            d["o_op"].Set(r)

# ---------------- right arm IK for the new phone pose (rigid move about the phone's start centre, weighted over the reach/carry) -------------
def sm(x): x = min(1.0, max(0.0, x)); return 0.5 - 0.5 * math.cos(math.pi * x)
def weight(f):
    if f <= REACH0: return 0.0
    if f < REACH1: return sm((f - REACH0) / (REACH1 - REACH0))
    if f <= LIFT_F: return 1.0
    if f < CARRY1: return 1.0 - sm((f - LIFT_F) / (CARRY1 - LIFT_F))
    return 0.0
c0 = P0[0][:3, 3].copy(); YAW = math.radians(a.dyaw)
def Tmove(w):
    M = np.eye(4); M[:3, :3] = rodr(np.array([0, 0, 1.0]), w * YAW); M[:3, 3] = c0 + w * np.array([a.dx, a.dy, 0]) - M[:3, :3] @ c0; return M
def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = math.acos(c)
    return np.zeros(3) if th < 1e-7 else th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
Qn = Q0.copy(); ik_pos, ik_rot = [], []
if a.dx or a.dy or a.dyaw:
    prev = None
    for f in range(T):
        w = weight(f)
        if w <= 0: prev = None; continue
        tgt = Tmove(w) @ fk(Q0[f]); q = Q0[f].copy()
        if prev is not None: q[R_ARM] = prev + (Q0[f, R_ARM] - Q0[f - 1, R_ARM])
        for it in range(40):
            M = fk(q); e = np.concatenate([tgt[:3, 3] - M[:3, 3], 0.3 * rotvec(tgt[:3, :3] @ M[:3, :3].T)])
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 1e-3: break
            Jm = np.zeros((6, 7))
            for k in range(7):
                qd_ = q.copy(); qd_[R_ARM[k]] += 1e-5; Md = fk(qd_)
                Jm[:3, k] = (Md[:3, 3] - M[:3, 3]) / 1e-5; Jm[3:, k] = 0.3 * rotvec(Md[:3, :3] @ M[:3, :3].T) / 1e-5
            q[R_ARM] = np.clip(q[R_ARM] + Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-5 * np.eye(6), e), LIM[R_ARM, 0], LIM[R_ARM, 1])
        M = fk(q); ik_pos.append(np.linalg.norm(tgt[:3, 3] - M[:3, 3])); ik_rot.append(np.linalg.norm(rotvec(tgt[:3, :3] @ M[:3, :3].T)))
        Qn[f] = q; prev = q[R_ARM].copy()
# phone: new table spot -> carried rigidly by the NEW hand -> recorded pose in the carton
Pn = P0.copy()
for f in range(T):
    if f < GRASP_F: Pn[f] = Tmove(1.0) @ P0[f]
    elif f < RELEASE_F: Pn[f] = fk(Qn[f]) @ np.linalg.inv(fk(Q0[f])) @ P0[f]
info = dict(dx=a.dx, dy=a.dy, dyaw=a.dyaw, ik_max_pos_mm=round(1000 * max(ik_pos, default=0), 2), ik_max_rot_deg=round(math.degrees(max(ik_rot, default=0)), 2),
            max_arm_step_rad=round(float(np.abs(np.diff(Qn[:, 2:16], axis=0)).max()), 3), limit_ok=bool(((Qn >= LIM[:, 0] - 1e-6) & (Qn <= LIM[:, 1] + 1e-6)).all()))
print("[gen] IK", json.dumps(info), flush=True)
if a.ik_only:
    json.dump(info, open(os.path.join(a.out, "variant.json"), "w"), indent=2); app.close(); raise SystemExit(0)

# ---------------- scene look: lights, table colour, phone print ----------------
for p in st.Traverse():
    ia = p.GetAttribute("inputs:intensity")
    if p.HasAPI(UsdLux.LightAPI) and ia and ia.Get() is not None and not str(p.GetPath()).startswith("/tomo_re_uhand"): ia.Set(float(ia.Get()) * a.light_gain)
UsdLux.DomeLight.Define(st, "/FS_Dome").CreateIntensityAttr(a.dome)
ROOM = os.path.join(ROOT, "assets", "room_backdrop.png")
if st.GetPrimAtPath("/root/Plane").IsValid() and os.path.exists(ROOM):   # room-photo backdrop: show the photo (exported material renders as fog)
    bm = UsdShade.Material.Define(st, "/FS_Looks/RoomMat"); bs = UsdShade.Shader.Define(st, "/FS_Looks/RoomMat/S"); bs.CreateIdAttr("UsdPreviewSurface")
    tx = UsdShade.Shader.Define(st, "/FS_Looks/RoomMat/Tex"); tx.CreateIdAttr("UsdUVTexture"); tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ROOM))
    rd = UsdShade.Shader.Define(st, "/FS_Looks/RoomMat/UV"); rd.CreateIdAttr("UsdPrimvarReader_float2"); rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    rgb_out = tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    bs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out); bs.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    bs.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0); bm.CreateSurfaceOutput().ConnectToSource(bs.ConnectableAPI(), "surface")
    for p in Usd.PrimRange(st.GetPrimAtPath("/root/Plane")):
        if p.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(bm, UsdShade.Tokens.strongerThanDescendants)
            print("[gen] room backdrop uv primvars", [pv.GetName() for pv in UsdGeom.PrimvarsAPI(p).GetPrimvars() if "st" in pv.GetName() or "UV" in pv.GetName()], flush=True)
if a.table_rgb:
    mat = UsdShade.Material.Define(st, "/FS_Looks/TableMat"); sh = UsdShade.Shader.Define(st, "/FS_Looks/TableMat/S"); sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, a.table_rgb.split(","))))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55); mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    for p in st.Traverse():
        if p.IsA(UsdGeom.Mesh) and "Alu_table" in str(p.GetPath()): UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
if a.phone_tex or a.phone_hue:
    for p in st.Traverse():
        if p.IsA(UsdShade.Shader) and "iPhone" in str(p.GetPath()):
            fi = UsdShade.Shader(p).GetInput("file")
            if fi and fi.Get() and "color" in str(fi.Get().path).lower():
                src = os.path.abspath(os.path.expanduser(a.phone_tex)) if a.phone_tex else str(fi.Get().resolvedPath or fi.Get().path)
                im = cv2.imread(src, cv2.IMREAD_COLOR)
                if a.phone_hue:
                    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); hsv[..., 0] = ((hsv[..., 0].astype(int) + int(a.phone_hue / 2)) % 180).astype(np.uint8); im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                dst = os.path.abspath(os.path.join(a.out, "phone_print.png")); cv2.imwrite(dst, im); fi.Set(Sdf.AssetPath(dst))
# phone xform: one 'fs' matrix op (session layer) overrides the exported animation
ph = st.GetPrimAtPath("/root/iPhone"); phx = UsdGeom.Xformable(ph); ph_op = phx.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "fs"); phx.SetXformOpOrder([ph_op])
SEALED_F = 445          # lid fully shut: the phone is sealed in the carton from here on (the .blend's two carton rigs let it poke out at the flip)
ph_vis = UsdGeom.Imageable(ph).GetVisibilityAttr()
# camera: user's intrinsics (same convention as runtime/isaac_runtime.py)
fx, fy, cx, cy, FW, FH = K; cam = UsdGeom.Camera(st.GetPrimAtPath(CAM)); hap = float(cam.GetHorizontalApertureAttr().Get() or 20.955)
foc = hap * fx / FW; vap = foc * FH / fy
cam.GetFocalLengthAttr().Set(foc); cam.GetHorizontalApertureAttr().Set(hap); cam.GetVerticalApertureAttr().Set(vap)
cam.GetHorizontalApertureOffsetAttr().Set((FW / 2 - cx) * hap / FW); cam.GetVerticalApertureOffsetAttr().Set((cy - FH / 2) * vap / FH)
cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
W = a.W or (1920 if a.cam == "robot" else 672); H = int(round(W * FH / FW))
rpd = rep.create.render_product(CAM, (W, H)); a_rgb = rep.AnnotatorRegistry.get_annotator("rgb"); a_dep = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
a_rgb.attach([rpd]); a_dep.attach([rpd])
tl = omni.timeline.get_timeline_interface(); tps = st.GetTimeCodesPerSecond() or 30.0

# ---------------- render ----------------
PW, PH = PACK
ys = (np.arange(PH) * H / PH + H / PH / 2).astype(int); xs = (np.arange(PW) * W / PW + W / PW / 2).astype(int)
dbg = [int(x) for x in a.frames.split(",")] if a.frames else None
flist = dbg or list(range(0, T, a.stride)); os.makedirs(RAW, exist_ok=True)
camw = np.zeros((T, 4, 4)); t0 = time.time()
for i, f in enumerate(flist):
    tl.set_current_time(f / tps); drive(Qn[f])
    par = np.array(UsdGeom.XformCache(Usd.TimeCode(f)).GetParentToWorldTransform(ph)).T
    ph_op.Set(Gf.Matrix4d(*(np.linalg.inv(par) @ Pn[f]).T.flatten().tolist()))
    ph_vis.Set(UsdGeom.Tokens.invisible if f >= SEALED_F else UsdGeom.Tokens.inherited)
    for _ in range(a.warm): app.update()
    rep.orchestrator.step(rt_subframes=4, pause_timeline=True)
    r = np.asarray(a_rgb.get_data()); d = np.asarray(a_dep.get_data())
    camw[f] = np.array(UsdGeom.XformCache(Usd.TimeCode(f)).GetLocalToWorldTransform(st.GetPrimAtPath(CAM))).T
    if dbg:
        cv2.imwrite(os.path.join(a.out, f"gen_{f:04d}.png"), cv2.cvtColor(r[..., :3], cv2.COLOR_RGB2BGR)); continue
    cv2.imwrite(os.path.join(RAW, f"rgb_{f:04d}.png"), cv2.cvtColor(r[..., :3], cv2.COLOR_RGB2BGR))
    dm = np.clip(np.rint(np.where(np.isfinite(d) & (d > 0) & (d < 60.0), d, 0.0) * 1000), 0, 65535).astype(np.uint16)   # no reading -> 0
    np.save(os.path.join(RAW, f"depth_{f:04d}.npy"), cv2.resize(dm, (PW, PH), interpolation=cv2.INTER_NEAREST))
    if i % 100 == 0: print(f"[gen] frame {f}/{T - 1}  {(time.time() - t0) / (i + 1):.2f}s/frame", flush=True)
info.update(cam=a.cam, K_full=list(K), pack_wh=list(PACK), render_s=round(time.time() - t0, 1), table_rgb=a.table_rgb, phone_tex=os.path.basename(a.phone_tex), phone_hue=a.phone_hue, renderer="isaac")
lab = {k: M0[k] for k in M0.files}; lab.update(q=Qn.astype(np.float32), phone_world=Pn, cam_world=camw)
np.savez_compressed(os.path.join(a.out, "labels_raw.npz"), **lab)
json.dump(info, open(os.path.join(a.out, "variant.json"), "w"), indent=2)
print("[gen] done", json.dumps(info), flush=True)
app.close()
