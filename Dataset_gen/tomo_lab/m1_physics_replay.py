"""Milestone 1: replay one recorded Find_and_Pick demo in Isaac Lab with PHYSICS ON.

The recordings were made with physics off (joints posed by forward kinematics, the phone carried by the suction
link). Here the robot is an articulation driven by PD joint targets (the recorded q46, 30 fps), the phone is a
rigid body and the suction is a surface gripper switched by the recorded io_events. Output: the head-camera
video, a per-frame log and how far the simulated phone ends from the recorded one - i.e. whether the demo still
works with real contact.

    ~/IsaacLab/isaaclab.sh -p m1_physics_replay.py --episode Find_and_Pick_001 --headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episode", default="Find_and_Pick_001")
parser.add_argument("--frames", type=int, default=0, help="replay only the first N frames (0 = all)")
parser.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs/m1")
parser.add_argument("--cam_every", type=int, default=2, help="save a head-camera frame every N recorded frames")
parser.add_argument("--arm_kp", type=float, default=1500.0)
parser.add_argument("--arm_kd", type=float, default=150.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.device = "cpu"                    # the surface gripper runs on the CPU physics pipeline only
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import (Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, SurfaceGripper,  # noqa: E402
                             SurfaceGripperCfg)
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from tomo_lab import scene as S  # noqa: E402
from tomo_lab.paths import (CANONICAL_46, SCENE_USD, URDF, EpisodeReference, RobotFK, episode_yaml,  # noqa: E402
                            isaac_config, pose_matrix)

ROOT = ""                              # the scene is a sublayer: its prims keep their own paths
FPS, SUBSTEPS = 30, 4                  # recorded frame rate; physics at 120 Hz


def log(*a):
    print("[m1]", *a, flush=True)


def urdf_root_link(urdf: Path) -> str:
    import xml.etree.ElementTree as ET
    r = ET.parse(str(urdf)).getroot()
    children = {j.find("child").get("link") for j in r.findall("joint")}
    return next(lk.get("name") for lk in r.findall("link") if lk.get("name") not in children)


def main():
    icfg = isaac_config()["isaac_mode"]
    ep = EpisodeReference(episode_yaml(args.episode))
    on, off = ep.suction_frames()
    T = ep.n_frames if not args.frames else min(args.frames, ep.n_frames)
    out = args.out / args.episode
    out.mkdir(parents=True, exist_ok=True)
    log(f"{ep.name}: {ep.n_frames} frames at {ep.fps} fps, suction on {on} off {off}; replaying {T}")

    sim = SimulationContext(SimulationCfg(dt=1.0 / (FPS * SUBSTEPS), render_interval=SUBSTEPS, device="cpu"))
    stage = omni.usd.get_context().get_stage()
    S.load_scene_sublayer(stage, SCENE_USD)
    log("extra physics scenes off:", S.deactivate_extra_physics_scenes(stage))
    log("scene fixes:", S.apply_scene_fixes(stage, ROOT, icfg.get("scene_fixes") or {}))

    S.move_articulation_root(stage, ROOT + icfg["robot_root"], ROOT + "/tomo_re_uhand")
    robot_path = ROOT + "/tomo_re_uhand"
    link_path = ROOT + icfg["suction_prim"]
    phone_path = ROOT + icfg["box_prim"]
    table = ROOT + (icfg.get("appearance") or {}).get("table_prim", "/Env/geo/TM1001_00_610001_R00")
    lo, hi = S.box_collider_for(stage, table, "/World/TableCollider")
    log(f"table collider: x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] top z {hi[2]:.3f}")

    # phone: the recordings give its box-centre frame; the rigid body is the /Iphone prim
    F0, size = S.phone_object_frame(stage, ROOT + icfg["phone_mesh_prim"])
    obj_to_prim = np.linalg.inv(F0) @ S.world(stage.GetPrimAtPath(phone_path))
    log(f"phone size (w, l, h) = {np.round(size, 4).tolist()} m")

    # suction cup: where the recorded phone top meets the suction link at the vacuum-on frame (URDF FK)
    fk = RobotFK(URDF)
    root_link = urdf_root_link(URDF)
    root_prim = next(p for p in stage.Traverse() if p.GetPath().pathString.startswith(robot_path)
                     and p.GetName() == root_link)
    T_link = S.world(root_prim) @ fk.link_transform(link_path.rsplit("/", 1)[1], dict(zip(CANONICAL_46, ep.q46(on))))
    top = pose_matrix(ep.phone_pose(on)) @ np.array([0, 0, size[2] / 2, 1.0])
    cup = (np.linalg.inv(T_link) @ top)[:3]
    cup_dir = cup / np.linalg.norm(cup)
    grip_path = S.add_suction_gripper(stage, link_path, cup - 0.004 * cup_dir, cup_dir)
    log(f"suction cup in {link_path.rsplit('/', 1)[1]} frame: {np.round(cup, 4).tolist()} m (root link {root_link})")

    # the robot is held by its controllers: fixed base, no gravity on its links, PD joint drives
    sim_utils.modify_rigid_body_properties(robot_path, sim_utils.RigidBodyPropertiesCfg(disable_gravity=True))
    sim_utils.modify_articulation_root_properties(
        robot_path, sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True, enabled_self_collisions=False))
    robot = Articulation(ArticulationCfg(prim_path=robot_path, spawn=None, actuators={
        "head": ImplicitActuatorCfg(joint_names_expr=["Head_joint.*"], stiffness=300.0, damping=30.0),
        "arms": ImplicitActuatorCfg(joint_names_expr=[".*_arm_joint.*"], stiffness=args.arm_kp, damping=args.arm_kd),
        "hands": ImplicitActuatorCfg(joint_names_expr=["[LR]F_joint_.*"], stiffness=60.0, damping=6.0),
    }))
    phone = RigidObject(RigidObjectCfg(prim_path=phone_path, spawn=None))
    grip = SurfaceGripper(SurfaceGripperCfg(prim_expr=grip_path, max_grip_distance=0.02, coaxial_force_limit=200.0,
                                            shear_force_limit=200.0, retry_interval=0.2))
    cam = Camera(CameraCfg(prim_path=ROOT + icfg["camera_prim"], spawn=None, width=480, height=360,
                           data_types=["rgb"], update_period=0))
    sim.reset()

    names = list(robot.joint_names)
    missing = [n for n in CANONICAL_46 if n not in names]
    log(f"robot: {len(names)} joints, {len(robot.body_names)} bodies; recorded joints missing: {missing}")
    idx = torch.tensor([names.index(n) for n in CANONICAL_46 if n in names])
    cols = [i for i, n in enumerate(CANONICAL_46) if n in names]
    arm = torch.tensor([names.index(n) for n in CANONICAL_46 if "_arm_joint" in n])

    def q_full(t):
        q = robot.data.default_joint_pos.clone()
        q[0, idx] = torch.tensor(ep.q46(t)[cols], dtype=q.dtype)
        return q

    q0 = q_full(0)
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    robot.set_joint_position_target(q0)
    M = pose_matrix(ep.phone_pose(0)) @ obj_to_prim
    pose = torch.tensor([[*M[:3, 3], *quat_from_matrix(torch.tensor(M[:3, :3])).tolist()]], dtype=torch.float32)
    phone.write_root_pose_to_sim(pose)
    phone.write_root_velocity_to_sim(torch.zeros(1, 6))
    for _ in range(10):                                     # settle
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        phone.update(sim.get_physics_dt())

    dt = sim.get_physics_dt()
    frames, rows = [], []
    for t in range(T):
        target = q_full(t)
        robot.set_joint_position_target(target)
        grip.set_grippers_command(torch.tensor([1.0 if on is not None and on <= t < (off or T) else -1.0]))
        for k in range(SUBSTEPS):
            robot.write_data_to_sim()
            grip.write_data_to_sim()
            sim.step(render=(k == SUBSTEPS - 1))
            robot.update(dt)
            phone.update(dt)
            grip.update(dt)
        P = np.eye(4)
        P[:3, 3] = phone.data.root_pos_w[0].numpy()
        from isaaclab.utils.math import matrix_from_quat
        P[:3, :3] = matrix_from_quat(phone.data.root_quat_w)[0].numpy()
        sim_obj = P @ np.linalg.inv(obj_to_prim)
        rec_obj = pose_matrix(ep.phone_pose(t))
        err = float(np.linalg.norm(sim_obj[:3, 3] - rec_obj[:3, 3]))
        track = float((robot.data.joint_pos[0, arm] - target[0, arm]).abs().max())
        rows.append({"t": t, "phone_xyz": np.round(sim_obj[:3, 3], 4).tolist(), "rec_xyz": np.round(rec_obj[:3, 3], 4).tolist(),
                     "phone_err_m": round(err, 4), "grip_state": int(grip.state[0]), "arm_track_err_rad": round(track, 4)})
        if t % args.cam_every == 0:
            cam.update(dt * SUBSTEPS)
            frames.append(cam.data.output["rgb"][0, ..., :3].numpy().astype(np.uint8))
        if t % 30 == 0 or t in (on, off):
            log(f"t={t:3d} grip={int(grip.state[0]):+d} phone_err={err:.3f} m z={sim_obj[2, 3]:.3f} "
                f"(rec {rec_obj[2, 3]:.3f}) arm_track={track:.3f} rad")

    z0 = rows[0]["phone_xyz"][2]
    zmax = max(r["phone_xyz"][2] for r in rows)
    summary = {"episode": ep.name, "frames": T, "suction": [on, off], "phone_lift_m": round(zmax - z0, 4),
               "phone_final_err_m": rows[-1]["phone_err_m"], "max_arm_track_err_rad": max(r["arm_track_err_rad"] for r in rows),
               "grip_closed_frames": sum(r["grip_state"] == 1 for r in rows)}
    (out / "log.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    if frames:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(out / "head_cam.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS / args.cam_every, (w, h))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
    log("SUMMARY", json.dumps(summary))
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception:                              # print it now: app.close() ends the process before Python would
        import traceback
        traceback.print_exc()
        raise
    finally:                                       # always shut Kit down, also after an error (else it hangs)
        sys.stdout.flush()
        app.close()
