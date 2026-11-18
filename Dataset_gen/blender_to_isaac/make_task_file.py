"""The training's task file for a generated dataset (generate.sh): scenes/<scene>.yaml task_yaml (+ task_files) copied
into the dataset; when the episodes are cut at a frame, only the steps and substeps they contain, and the robot's W2 + IK
lines only while their steps are still there (the training refuses a task.yaml that names a step the episodes lack).

    env/bin/python make_task_file.py <scene> <dataset dir> <last frame, -1 = whole episodes>
"""
import json
import shutil
import sys
from pathlib import Path

import yaml

from scene_robot import recorder_steps

HERE = Path(__file__).resolve().parent


def scene_steps(cfg, scene):
    """[[step, first frame], ...]: the scene file's steps, else the recorder's sub-episodes in the .blend, else START + one step per
    marker - the same steps pack_episode.py labels the episodes with (make_motion.py reads them the same way)."""
    if cfg.get("steps"):
        return cfg["steps"]
    doc = json.loads((HERE / "blender_export" / scene / "rig.json").read_text())
    if recorder_steps(doc):
        return recorder_steps(doc)
    markers = doc["scene_mmrt"].get("mmrt_markers") or doc["markers"]
    return [["START", 0]] + [[m.get("marker_name") or m["name"], m["frame"]] for m in markers]


def cut_robot(robot, step_names, names):
    """The task's robot lines (w2_grab / w2_drop: a step, grab_check: a step or substep) that the cut file still has.
    W2 + IK needs both its grab and its drop step, so without one of them none of it is kept."""
    keep = {k: v for k, v in robot.items() if v in (names if k == "grab_check" else step_names)}
    w2 = {"w2_grab", "w2_drop"}
    return {} if w2 & set(robot) and not w2 <= set(keep) else keep


def main():
    scene, out, last = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    cfg = yaml.safe_load((HERE / "scenes" / f"{scene}.yaml").read_text())
    for f in cfg.get("task_files", []):
        shutil.copy(HERE / f, out / Path(f).name)
    if not cfg.get("task_yaml"):
        return
    src = HERE / cfg["task_yaml"]
    if last < 0:
        shutil.copy(src, out / "task.yaml")
        return
    t = yaml.safe_load(src.read_text())
    keep = {s for s, f in scene_steps(cfg, scene) if f <= last}
    t["steps"] = [s for s in t["steps"] if s["name"] in keep]
    t["substeps"] = [s for s in t.get("substeps") or [] if (s.get("frames") or [0, 0])[1] <= last]
    names = keep | {s["name"] for s in t["substeps"]}
    if t.get("readouts"):
        t["readouts"]["map"] = {k: v for k, v in (t["readouts"].get("map") or {}).items() if k in names}
    if t.get("robot"):
        step_names = {s["name"] for s in t["steps"]}
        robot = cut_robot(t["robot"], step_names, step_names | {s["name"] for s in t["substeps"]})
        if robot:
            t["robot"] = robot
        else:
            del t["robot"]
    (out / "task.yaml").write_text(f"# {src.name} cut at frame {last} by generate.sh: only the steps the episodes "
                                   f"contain\n" + yaml.safe_dump(t, sort_keys=False, width=120))


if __name__ == "__main__":
    main()
