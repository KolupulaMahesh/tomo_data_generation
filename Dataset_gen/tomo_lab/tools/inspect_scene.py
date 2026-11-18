"""Print what the tomo packing scene USD contains for physics: articulation roots, joints (by type and owner),
rigid bodies, colliders, the carton's sub-parts and the suction / camera prims.

    ~/IsaacLab/isaaclab.sh -p tools/inspect_scene.py <scene.usd>
"""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import collections  # noqa: E402

from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(sys.argv[1])
roots, joints, bodies, colliders = [], collections.Counter(), [], 0
joint_owner = collections.Counter()
for p in stage.Traverse():
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        roots.append(str(p.GetPath()))
    if p.IsA(UsdPhysics.Joint):
        joints[p.GetTypeName()] += 1
        joint_owner["/".join(str(p.GetPath()).split("/")[:3])] += 1
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        bodies.append(str(p.GetPath()))
    if p.HasAPI(UsdPhysics.CollisionAPI):
        colliders += 1
print("articulation roots:", roots)
print("joints by type:", dict(joints))
print("joints by owner:", dict(joint_owner))
print("rigid bodies:", len(bodies), [b for b in bodies if "tomo_re_uhand/Geometry" not in b][:12])
print("colliders:", colliders)
for path in ("/tomo_re_uhand/Box_1_v1___open", "/Iphone"):
    p = stage.GetPrimAtPath(path)
    if p:
        kids = [c.GetName() for c in Usd.PrimRange(p) if c != p][:25]
        print(f"{path}: type {p.GetTypeName()}, rigid {p.HasAPI(UsdPhysics.RigidBodyAPI)}, children {kids}")
sys.stdout.flush()
app.close()
