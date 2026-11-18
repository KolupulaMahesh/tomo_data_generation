"""Find_and_Pick task terms: suction action, observations, subtask signal, success, phone reset."""
from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from .assets import PHONE_SIZE, PHONE_START, TABLE_TOP_Z, TRAY

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ------------------------------------------------------------------------------------------------ suction action
class SuctionAction(ActionTerm):
    """Sticky suction. One value per env: > 0 = on, <= 0 = off. While on, the phone is picked up as soon as its
    top is within cfg.attach_dist of the cup, then follows the cup rigidly (pose written every physics step);
    switching off releases it to physics (it falls / settles). No surface-gripper extension needed."""

    cfg: SuctionActionCfg

    def __init__(self, cfg: SuctionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)              # cfg.asset_name = the robot
        from .assets import CUP_IN_LINK, CUP_LINK
        self._phone = env.scene[cfg.object_name]
        self._link = self._asset.body_names.index(CUP_LINK)
        self._off = torch.tensor(CUP_IN_LINK, device=self.device).expand(self.num_envs, 3)
        self._top = torch.tensor([0.0, 0.0, PHONE_SIZE[2] / 2], device=self.device).expand(self.num_envs, 3)
        self._raw = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd = -torch.ones(self.num_envs, device=self.device)
        self.attached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rel_p = torch.zeros(self.num_envs, 3, device=self.device)
        self._rel_q = torch.zeros(self.num_envs, 4, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._cmd.unsqueeze(-1)

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        self._cmd = torch.where(actions[:, 0] > 0.0, 1.0, -1.0)

    def apply_actions(self):                    # called every physics step
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
        d = self._asset.data
        cup_p, cup_q = combine_frame_transforms(d.body_pos_w[:, self._link], d.body_quat_w[:, self._link], self._off)
        ph_p, ph_q = self._phone.data.root_pos_w, self._phone.data.root_quat_w
        on = self._cmd > 0
        top, _ = combine_frame_transforms(ph_p, ph_q, self._top)
        grab = on & ~self.attached & ((top - cup_p).norm(dim=-1) < self.cfg.attach_dist)
        if grab.any():
            rp, rq = subtract_frame_transforms(cup_p[grab], cup_q[grab], ph_p[grab], ph_q[grab])
            self._rel_p[grab], self._rel_q[grab] = rp, rq
        self.attached = (self.attached | grab) & on
        ids = self.attached.nonzero(as_tuple=False).squeeze(-1)
        if len(ids):
            p, q = combine_frame_transforms(cup_p[ids], cup_q[ids], self._rel_p[ids], self._rel_q[ids])
            self._phone.write_root_pose_to_sim(torch.cat([p, q], dim=-1), env_ids=ids)
            self._phone.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    def reset(self, env_ids=None) -> None:
        ids = env_ids if env_ids is not None else slice(None)
        self._raw[ids] = 0.0
        self._cmd[ids] = -1.0
        self.attached[ids] = False


@configclass
class SuctionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SuctionAction
    asset_name: str = "robot"
    object_name: str = "phone"
    attach_dist: float = 0.025                  # cup within 2.5 cm of the phone's top centre -> picked up


# ------------------------------------------------------------------------------------------------ observations
def phone_pose(env: ManagerBasedRLEnv, cfg: SceneEntityCfg = SceneEntityCfg("phone")) -> torch.Tensor:
    obj = env.scene[cfg.name]
    return torch.cat([obj.data.root_pos_w - env.scene.env_origins, obj.data.root_quat_w], dim=-1)


def cup_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Suction-cup frame (RF_link_52 + the cup offset) in the env frame: pos (3) + quat wxyz (4) - the Mimic eef."""
    from isaaclab.utils.math import combine_frame_transforms

    from .assets import CUP_IN_LINK, CUP_LINK
    robot = env.scene["robot"]
    if not hasattr(env, "_tomo_cup_idx"):
        env._tomo_cup_idx = robot.body_names.index(CUP_LINK)
    i = env._tomo_cup_idx
    off = torch.tensor(CUP_IN_LINK, device=env.device).expand(env.num_envs, 3)
    pos, quat = combine_frame_transforms(robot.data.body_pos_w[:, i], robot.data.body_quat_w[:, i], off)
    return torch.cat([pos - env.scene.env_origins, quat], dim=-1)


def cup_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    return cup_pose(env)[:, :3]


def cup_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    return cup_pose(env)[:, 3:]


def _holding(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.action_manager.get_term("suction_action").attached


def suction_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 holding the phone, -1 not."""
    return torch.where(_holding(env), 1.0, -1.0).unsqueeze(-1)


def phone_grasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mimic subtask signal 'grasp': the suction holds the phone and it is off the table."""
    z = env.scene["phone"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (_holding(env) & (z > TABLE_TOP_Z + PHONE_SIZE[2] / 2 + 0.01)).float()


def phone_in_tray(env: ManagerBasedRLEnv) -> torch.Tensor:
    p = env.scene["phone"].data.root_pos_w - env.scene.env_origins
    return ((p[:, 0] - TRAY["cx"]).abs() < TRAY["lx"] / 2) & ((p[:, 1] - TRAY["cy"]).abs() < TRAY["ly"] / 2) & \
        (p[:, 2] < TRAY["floor"] + TRAY["wall"])


# ------------------------------------------------------------------------------------------------ termination
def placed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Success: the phone lies in the carton tray, released and at rest."""
    still = env.scene["phone"].data.root_lin_vel_w.norm(dim=-1) < 0.05
    return phone_in_tray(env) & ~_holding(env) & still


# ------------------------------------------------------------------------------------------------ events
def reset_phone(env: ManagerBasedRLEnv, env_ids: torch.Tensor, override_attr: str = "tomo_phone_override"):
    """Phone flat on the table at a random pose within the recordings' range - or, if the script set
    env.<override_attr> = (x, y, yaw), exactly there (seed-demo replay)."""
    n = len(env_ids)
    dev = env.device
    override = getattr(env, override_attr, None)
    if override is not None:
        x, y, yaw = (torch.full((n,), float(v), device=dev) for v in override)
    else:
        def u(lo_hi):
            return torch.empty(n, device=dev).uniform_(*lo_hi)
        x, y, yaw = u(PHONE_START["x"]), u(PHONE_START["y"]), u(PHONE_START["yaw"])
    pos = torch.stack([x, y, torch.full_like(x, TABLE_TOP_Z + PHONE_SIZE[2] / 2 + 0.001)], dim=-1)
    quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    phone = env.scene["phone"]
    phone.write_root_pose_to_sim(torch.cat([pos + env.scene.env_origins[env_ids], quat], dim=-1), env_ids=env_ids)
    phone.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


def yaw_deg(q: torch.Tensor) -> float:
    return math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
