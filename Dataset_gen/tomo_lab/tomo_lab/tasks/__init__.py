"""Gym registration of the tomo tasks (pure Python: safe to import before the simulator starts)."""
import gymnasium as gym

gym.register(
    id="Tomo-FindPick-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "tomo_lab.tasks.findpick_env_cfg:TomoFindPickEnvCfg"},
    disable_env_checker=True,
)
gym.register(
    id="Tomo-FindPick-IK-Abs-Mimic-v0",
    entry_point="tomo_lab.tasks.findpick_mimic_env:TomoFindPickMimicEnv",
    kwargs={"env_cfg_entry_point": "tomo_lab.tasks.findpick_mimic_env_cfg:TomoFindPickMimicEnvCfg"},
    disable_env_checker=True,
)
