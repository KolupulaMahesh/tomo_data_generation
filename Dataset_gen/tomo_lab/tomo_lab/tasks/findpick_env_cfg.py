"""Tomo Find_and_Pick with physics: pick the phone box with the right hand's suction cup, put it in the carton.

Action (8): suction-cup pose in the env frame, absolute (pos 3 + quat wxyz 4), solved by differential IK on the
seven right-arm joints, + suction (> 0 on). Head, left arm and fingers hold the grasp posture. Modelled on
Isaac Lab's Isaac-Stack-Cube-Franka-IK-Abs-v0 (same action layout, same Mimic hooks).
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .. import mdp
from ..assets import (CARTON_POSE, CARTON_USD, CUP_IN_LINK, CUP_LINK, PHONE_SIZE, ROOM_POSE, ROOM_USD, TABLE_TOP_Z,
                      TOMO_CFG, TRAY)

NO_COLLISION = sim_utils.CollisionPropertiesCfg(collision_enabled=False)


def _tray_wall(name, size, center, visible=False):
    """Static, invisible box collider directly under the env prim (a regex spawn needs an existing parent)."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Col_" + name,
        spawn=sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(), visible=visible),
        init_state=AssetBaseCfg.InitialStateCfg(pos=center))


T = TRAY
_wall_z = T["floor"] + T["wall"] / 2


@configclass
class FindPickSceneCfg(InteractiveSceneCfg):
    robot = TOMO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    phone = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Phone",
        spawn=sim_utils.CuboidCfg(
            size=PHONE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.93, 0.93, 0.95)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.54, -0.41, TABLE_TOP_Z + PHONE_SIZE[2] / 2 + 0.001)),
    )
    # looks: the dataset's room/table and carton (no collision); physics: simple boxes below
    room = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Room",
                        spawn=sim_utils.UsdFileCfg(usd_path=str(ROOM_USD), scale=(0.01, 0.01, 0.01),
                                                   collision_props=NO_COLLISION),
                        init_state=AssetBaseCfg.InitialStateCfg(pos=ROOM_POSE["pos"], rot=ROOM_POSE["rot"]))
    carton = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Carton",
                          spawn=sim_utils.UsdFileCfg(usd_path=str(CARTON_USD), scale=(0.01, 0.01, 0.01),
                                                     collision_props=NO_COLLISION),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=CARTON_POSE["pos"], rot=CARTON_POSE["rot"]))
    table_collider = _tray_wall("table", (1.0, 1.0, 0.04), (0.375, 0.0, TABLE_TOP_Z - 0.02))
    tray_floor = _tray_wall("tray_floor", (T["lx"], T["ly"], 0.006), (T["cx"], T["cy"], T["floor"] - 0.003))
    tray_wall_near = _tray_wall("tray_near", (0.006, T["ly"], T["wall"]), (T["cx"] - T["lx"] / 2, T["cy"], _wall_z))
    tray_wall_far = _tray_wall("tray_far", (0.006, T["ly"], T["wall"]), (T["cx"] + T["lx"] / 2, T["cy"], _wall_z))
    tray_wall_left = _tray_wall("tray_left", (T["lx"], 0.006, T["wall"]), (T["cx"], T["cy"] + T["ly"] / 2, _wall_z))
    tray_wall_right = _tray_wall("tray_right", (T["lx"], 0.006, T["wall"]), (T["cx"], T["cy"] - T["ly"] / 2, _wall_z))
    tray_under = _tray_wall("tray_under", (T["lx"], T["ly"], T["floor"]), (T["cx"], T["cy"], T["floor"] / 2))
    # the head camera with the training intrinsics (fx 606.4 at 960x720 = 13.24 mm on a 20.955 mm aperture)
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_camera_link/head_cam",
        height=360, width=480, data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.2365, horizontal_aperture=20.955, clipping_range=(0.01, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 1.0, 0.0, 0.0), convention="opengl"),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))


@configclass
class ActionsCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot", joint_names=["Right_arm_joint[1-7]"], body_name=CUP_LINK,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=CUP_IN_LINK),
    )
    suction_action = mdp.SuctionActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=base_mdp.last_action)
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.cup_pos)
        eef_quat = ObsTerm(func=mdp.cup_quat)
        phone_pose = ObsTerm(func=mdp.phone_pose)
        suction = ObsTerm(func=mdp.suction_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp = ObsTerm(func=mdp.phone_grasped)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.placed)


@configclass
class EventCfg:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_phone = EventTerm(func=mdp.reset_phone, mode="reset")


@configclass
class TomoFindPickEnvCfg(ManagerBasedRLEnvCfg):
    scene: FindPickSceneCfg = FindPickSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4                   # 30 Hz control (the recordings' rate), 120 Hz physics
        self.episode_length_s = 12.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.device = "cpu"               # the surface gripper runs on the CPU physics pipeline only
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.viewer.eye = (1.6, -1.2, 1.1)
        self.viewer.lookat = (0.4, -0.1, 0.1)
