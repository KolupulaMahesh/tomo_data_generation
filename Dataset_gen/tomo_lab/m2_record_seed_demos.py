"""Milestone 2: turn the recorded Find_and_Pick demos into Isaac Lab demos WITH PHYSICS (Mimic's input).

Each recording (q46 at 30 fps, phone start pose, suction on 155 / off 290) becomes actions for the env:
the suction-cup pose from forward kinematics of the recorded joints (absolute, env frame) + the suction switch.
They are executed with physics (IK on the right arm, surface gripper, rigid phone); only demos that really
end with the phone released in the carton tray are exported - like Isaac Lab's record_demos.py.

    ~/IsaacLab/isaaclab.sh -p m2_record_seed_demos.py --episodes all --headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--episodes", default="Find_and_Pick_001", help="comma list, or 'all'")
parser.add_argument("--dataset_file", type=Path, default=Path(__file__).parent / "datasets/findpick_seed.hdf5")
parser.add_argument("--num_success_steps", type=int, default=5)
parser.add_argument("--video", action="store_true", help="save the head camera of each demo to outputs/m2/")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.device = "cpu"
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import tomo_lab.tasks  # noqa: E402,F401
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from tomo_lab.assets import CUP_IN_LINK, CUP_LINK, SEED_START, SUCTION_OFF, SUCTION_ON  # noqa: E402
from tomo_lab.paths import CANONICAL_46, FINDPICK_DIR, URDF, EpisodeReference, RobotFK  # noqa: E402

TASK = "Tomo-FindPick-IK-Abs-v0"


def log(*a):
    print("[m2]", *a, flush=True)


def seed_actions(ep: EpisodeReference, fk: RobotFK) -> torch.Tensor:
    """Per recorded frame from SEED_START: [cup pos (3), cup quat wxyz (4), suction]."""
    rows = []
    cup = np.array([*CUP_IN_LINK, 1.0])
    for t in range(SEED_START, ep.n_frames):
        T = fk.link_transform(CUP_LINK, dict(zip(CANONICAL_46, ep.q46(t))))   # robot base = env frame
        q = quat_from_matrix(torch.tensor(T[:3, :3], dtype=torch.float32))
        rows.append([*(T @ cup)[:3], *q.tolist(), 1.0 if SUCTION_ON <= t < SUCTION_OFF else -1.0])
    return torch.tensor(rows, dtype=torch.float32)


def main():
    names = sorted(d.name for d in FINDPICK_DIR.glob("Find_and_Pick_*")) if args.episodes == "all" \
        else args.episodes.split(",")
    env_cfg = parse_env_cfg(TASK, device="cpu", num_envs=1)
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    args.dataset_file.parent.mkdir(parents=True, exist_ok=True)
    env_cfg.recorders.dataset_export_dir_path = str(args.dataset_file.parent)
    env_cfg.recorders.dataset_filename = args.dataset_file.stem
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    fk = RobotFK(URDF)
    out = Path(__file__).parent / "outputs/m2"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        ep = EpisodeReference(FINDPICK_DIR / name / f"{name}.mmrt.yaml")
        p0 = ep.phone_pose(0)
        env.tomo_phone_override = (p0[0], p0[1], p0[5])             # the recording's phone start (x, y, yaw)
        env.recorder_manager.reset()
        env.reset()
        actions = seed_actions(ep, fk)
        ok_steps, success, frames, worst_cup = 0, False, [], 0.0
        for k, a in enumerate(actions):
            obs, *_ = env.step(a.unsqueeze(0))
            worst_cup = max(worst_cup, float((obs["policy"]["eef_pos"][0] - a[:3]).norm()))
            if k % 30 == 0:
                ph = env.scene["phone"].data.root_pos_w[0]
                log(f"{name} step {k}/{len(actions)} cup_err {float((obs['policy']['eef_pos'][0] - a[:3]).norm()):.3f} m "
                    f"holding {int(obs['policy']['suction'][0, 0])} phone z {float(ph[2]):.3f}")
            if args.video and k % 2 == 0:
                frames.append(env.scene["head_cam"].data.output["rgb"][0, ..., :3].numpy().astype(np.uint8))
            if bool(success_term.func(env, **success_term.params)[0]):
                ok_steps += 1
                if ok_steps >= args.num_success_steps:
                    success = True
                    break
            else:
                ok_steps = 0
        if not success:                       # keep stepping a little with the suction off, the phone may settle
            for _ in range(30):
                env.step(actions[-1:].clone())
                if bool(success_term.func(env, **success_term.params)[0]):
                    ok_steps += 1
                    if ok_steps >= args.num_success_steps:
                        success = True
                        break
        phone = env.scene["phone"].data.root_pos_w[0].numpy() - env.scene.env_origins[0].numpy()
        if success:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool))
            env.recorder_manager.export_episodes([0])
        r = {"episode": name, "success": success, "steps": k + 1, "phone_end": np.round(phone, 3).tolist(),
             "max_cup_track_err_m": round(worst_cup, 4)}
        results.append(r)
        log(json.dumps(r))
        if frames:
            import cv2
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
            for f in frames:
                vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            vw.release()
    n_ok = sum(r["success"] for r in results)
    log(f"SUMMARY {n_ok}/{len(results)} recordings succeed with physics -> {args.dataset_file}")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:                              # print it now: app.close() ends the process before Python would
        import traceback
        traceback.print_exc()
        raise
    finally:
        sys.stdout.flush()
        app.close()
