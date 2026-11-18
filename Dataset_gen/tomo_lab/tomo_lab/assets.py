"""Tomo robot + packing scene assets for Isaac Lab (physics ON). Geometry measured from the dataset scene
(tomo_packing_suction.usd, world = robot base frame, table top at z = 0) and the 46 Find_and_Pick recordings."""
from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.utils import configclass

from .paths import CANONICAL_46, FINDPICK_DIR, ISAAC_CODE, EpisodeReference

HERE = Path(__file__).resolve().parents[1]
TOMO_USD = HERE / "assets/tomo/tomo.usd"                       # NVIDIA URDF importer output (fixed base)
ROOM_USD = ISAAC_CODE / "scene/0/_apple_packing_env_z.usd"     # table, jig, walls (looks only)
CARTON_USD = ISAAC_CODE / "scene/Box 1_v1 - open/Box 1_v1 - open.usd"

# suction cup on RF_link_52: where the recorded phone top meets it at vacuum-on (identical in all 46 recordings)
CUP_LINK = "RF_link_52"
CUP_IN_LINK = (0.1203, -0.0139, -0.0455)
CUP_DIR_IN_LINK = (0.8578, -0.5140, 0.0)                       # the cup faces world -z at the grasp
PHONE_SIZE = (0.1795, 0.0953, 0.029)                           # x = long axis (the recordings' box-centre frame)
PHONE_START = {"x": (0.445, 0.639), "y": (-0.514, -0.316), "yaw": (math.radians(46), math.radians(128))}
TABLE_TOP_Z = 0.0
TRAY = {"cx": 0.41, "cy": 0.0, "lx": 0.155, "ly": 0.277, "floor": 0.035, "wall": 0.04}   # carton tray (after fix)
CARTON_POSE = {"pos": (0.4091, 0.0, 0.0348), "rot": (-0.5, -0.5, 0.5, 0.5)}           # scene pose - 6.36 cm in x
ROOM_POSE = {"pos": (0.0, 0.0, 0.0), "rot": (0.0, 0.0, 0.7071068, 0.7071068)}
SEED_START = 90                                               # recordings: 0-90 look around, reach from 91
SUCTION_ON, SUCTION_OFF = 155, 290


def grasp_posture(episode: str = "Find_and_Pick_001", frame: int = SUCTION_ON) -> dict[str, float]:
    """Start posture: head, left arm and fingers as at the grasp (they are held there); the right arm as at
    SEED_START, where every recording (all 46 identical) begins its reach. The right arm then follows the IK."""
    ep = EpisodeReference(FINDPICK_DIR / episode / f"{episode}.mmrt.yaml")
    q = {n: float(v) for n, v in zip(CANONICAL_46, ep.q46(frame))}
    q.update({n: float(v) for n, v in zip(CANONICAL_46, ep.q46(SEED_START)) if n.startswith("Right_arm_joint")})
    return q


def spawn_tomo_with_suction(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the robot, then add a surface (suction) gripper at the cup on every robot instance."""
    import omni.usd
    from isaaclab.sim.utils import find_matching_prim_paths

    from . import scene as S

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = omni.usd.get_context().get_stage()
    for path in find_matching_prim_paths(prim_path):
        if not stage.GetPrimAtPath(f"{path}/{CUP_LINK}/SurfaceGripper").IsValid():
            S.add_suction_gripper(stage, f"{path}/{CUP_LINK}", CUP_IN_LINK, CUP_DIR_IN_LINK, body1=f"{path}/RF_link_51")
    return prim


@configclass
class TomoUsdFileCfg(sim_utils.UsdFileCfg):
    func = spawn_tomo_with_suction


TOMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(                 # suction is the sticky SuctionAction (mdp.py), no gripper prim
        usd_path=str(TOMO_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos=grasp_posture()),
    actuators={
        "head": ImplicitActuatorCfg(joint_names_expr=["Head_joint.*"], stiffness=300.0, damping=30.0),
        "arms": ImplicitActuatorCfg(joint_names_expr=[".*_arm_joint.*"], stiffness=1500.0, damping=150.0),
        "hands": ImplicitActuatorCfg(joint_names_expr=["[LR]F_joint_.*"], stiffness=80.0, damping=8.0),
    },
)
