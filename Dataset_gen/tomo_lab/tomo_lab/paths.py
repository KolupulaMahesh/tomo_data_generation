"""Where the tomo assets and recordings live, and the trainer repo helpers this package reuses (pure numpy)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

FROZEN = Path(__file__).resolve().parents[1] / "frozen"             # copies taken 2026-11-03: this folder needs nothing from the brain
ISAAC_CODE = FROZEN / "isaac_code"
SCENE_USD = ISAAC_CODE / "scene/tomo_packing_suction.usd"           # robot articulation + table + phone + carton
CONFIG_YAML = ISAAC_CODE / "config.yaml"                             # scene_fixes, camera prim, suction link, ...
FINDPICK_DIR = ISAAC_CODE / "dataset/findandpick_new"                # Find_and_Pick_XXX/*.mmrt.yaml recordings
URDF = FROZEN / "full_robot.urdf"
FOLDABLE_CARTON_USD = FROZEN / "cardboard_dim/Box_Foldable_Physics.usda"   # tray + lid + 2 flaps, hinged


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the recorded-episode reader and the Blender pose convention of the existing (physics-off) runtime
_rt = _load_module("tomo_isaac_runtime", ISAAC_CODE / "runtime/isaac_runtime.py")
EpisodeReference = _rt.EpisodeReference
CANONICAL_46 = list(_rt.CANONICAL_46)
pose_matrix = _rt._pose_matrix          # (x, y, z, rx, ry, rz) Blender XYZ Euler -> 4x4
rot_xyz = _rt._rot_xyz

sys.path.insert(0, str(FROZEN))
from kinematics.fk import RobotFK  # noqa: E402  (URDF forward kinematics, numpy)


def isaac_config() -> dict:
    return yaml.safe_load(CONFIG_YAML.read_text())


def episode_yaml(name: str) -> Path:
    return FINDPICK_DIR / name / f"{name}.mmrt.yaml"
