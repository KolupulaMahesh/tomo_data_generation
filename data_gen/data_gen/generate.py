"""Generate a recovery dataset.

    cd data_gen
    python3 -m data_gen.generate --episodes 20 --out out/test            # from the MMRT recording (phone scenarios)
    python3 -m data_gen.generate --episodes 50 --motion-dir ../Dataset_gen/blender_to_isaac/runs/gidora_packing \
        --out out/gidora --render                                          # full scene, rendered in Isaac Sim

--motion-dir: blender_to_isaac runs of `make_motion.py gidora_packing --seed N` (ep_NNNN/generic/motion.npz). Each episode
uses the next one, so phone / object positions and looks differ per episode (category D). Without it, the MMRT
recording is the only demo (no rendering, no carton / tape scenarios).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from . import carry_fix, episode, export, source, stats, validate
from .detect import Grasp
from .kin import Kin

HERE = Path(__file__).resolve().parents[1]


def load_cfg(task_path, fail_path):
    task = yaml.safe_load(Path(task_path).read_text())
    base = Path(task_path).parent
    for k in ("mmrt", "urdf", "motion"):
        if task["source"].get(k):
            task["source"][k] = str((base / task["source"][k]).resolve())
    return task, yaml.safe_load(Path(fail_path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=str(HERE / "config/task_gidora.yaml"))
    ap.add_argument("--failures", default=str(HERE / "config/failures.yaml"))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "out/test"))
    ap.add_argument("--category", choices=["normal", "single", "multi", "unrecoverable"], default=None)
    ap.add_argument("--scenario", default=None, help="only this scenario (failures.yaml name), for testing")
    ap.add_argument("--scenarios", default=None, help="exactly these scenarios in each episode (comma list)")
    ap.add_argument("--motion", default=None, help="this one demo motion.npz (e.g. runs/gidora_packing/ep_0004/...)")
    ap.add_argument("--motion-dir", default=None)
    ap.add_argument("--scene", default="gidora_packing")
    ap.add_argument("--preview", action="store_true", help="write preview.mp4 per episode")
    ap.add_argument("--render", action="store_true", help="render in Isaac Sim + pack to the training format")
    a = ap.parse_args()

    task, fcfg = load_cfg(a.task, a.failures)
    if a.scenario:
        for k in fcfg["scenarios"]:
            fcfg["scenarios"][k]["weight"] = 1 if k == a.scenario else 0
    if a.scenarios:
        want = a.scenarios.split(",")
        for k in fcfg["scenarios"]:
            fcfg["scenarios"][k]["weight"] = 1 if k in want else 0
        fcfg["multi_count"] = [len(want), len(want)]
    motions = [Path(a.motion)] if a.motion else \
        sorted(Path(a.motion_dir).glob("ep_*/generic/motion.npz")) if a.motion_dir else []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kin = None
    rows = []
    for k in range(a.episodes):
        seed = a.seed0 + k
        mpath = motions[k % len(motions)] if motions else None
        src = source.load(task, mpath)
        if kin is None or kin.joints != src.joints:
            kin = Kin(task["source"]["urdf"], src.joints)
        carry = carry_fix.fix(src, task, kin)                       # the demo's own carry >= carton_gap_m (user 2026-11-18)
        t0 = time.time()
        grasps = {kk: Grasp(kk, v, src, kin, task["hands"]) for kk, v in task["grasps"].items()}
        for redraw in range(6):                                     # ROBOT LIMITS: an episode with any joint outside the
            draw = seed + 100000 * redraw                           # limits (or a generated one closer than 2 deg) is
            try:
                b, meta = episode.run(task, fcfg, src, kin, draw, a.category)   # never kept: drawn again
            except RuntimeError as err:                                 # e.g. no landing in view / off the stands
                lim_ok = False
                print(f"[gen] seed {seed} draw {redraw}: {err} -> drawn again", flush=True)
                continue
            meta["source"] = str(mpath or task["source"]["mmrt"])
            val = validate.run(b, task, fcfg["tolerances"], grasps, meta)
            lim_ok = val["checks"]["joint_limits"]["ok"] and val["checks"]["joint_limit_margin"]["ok"] \
                and val["checks"]["arm_clearance"]["ok"] \
                and all(v["ok"] for k, v in val["checks"].items() if k.endswith("_carton_gap"))   # too close to the carton
            if lim_ok:
                break
            print(f"[gen] seed {seed} draw {redraw}: robot limits / arm gap not kept "
                  f"{val['checks']['joint_limits']['outside'][:2]} {val['checks']['joint_limit_margin']['too_close'][:2]} "
                  f"gap {val['checks']['arm_clearance']['min_gap_mm']} mm "
                  f"{ {k: v['min_gap_mm'] for k, v in val['checks'].items() if k.endswith('_carton_gap')} } -> drawn again", flush=True)
        meta["draw"] = int(redraw)
        if not lim_ok:
            print(f"[gen] seed {seed}: no valid episode after 6 draws - episode SKIPPED", flush=True)
            continue
        name = f"ep_{seed:04d}"
        ed = out / name
        export.write_labels(b, ed, meta, val)
        if a.preview:
            export.preview(b, ed / "preview.mp4", task)
        if a.render:
            if not mpath:
                raise SystemExit("--render needs --motion-dir (the full scene from blender_to_isaac)")
            render(b, mpath, ed, out, seed, a.scene)
        rows.append({"episode": name, **meta, "frames": b.n, "success": val["success"],
                     "failed_checks": val["failed_checks"], "seconds": round(time.time() - t0, 1)})
        print(f"[gen] {name} {meta['category']:13s} {','.join(s['name'] for s in meta['scenarios']) or '-':40s} "
              f"frames {b.n:5d} success {val['success']} {val['failed_checks'] or ''}", flush=True)
    (out / "episodes.json").write_text(json.dumps(rows, indent=1))
    st = stats.summarize(out)
    print(json.dumps(st["totals"], indent=1))


def render(b, mpath, ed, out, seed, scene):
    b2i = export.B2I
    run_dir = ed / "isaac"
    info = export.write_isaac_episode(b, mpath, run_dir, scene)
    (ed / "isaac_export.json").write_text(json.dumps(info))
    subprocess.run([str(Path.home() / "isaacsim/python.sh"), str(b2i / "render_episode.py"), str(run_dir)], check=True)
    fs = out / "dataset" / f"fs_{seed:03d}"
    subprocess.run([str(b2i / "env/bin/python"), str(b2i / "pack_episode.py"), str(run_dir), str(fs)], check=True)
    for f in ("recovery_labels.npz", "events.json", "validation.json"):
        shutil.copy(ed / f, fs / f)
    shutil.rmtree(run_dir / "raw", ignore_errors=True)


if __name__ == "__main__":
    main()
