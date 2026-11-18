"""Which robot a scene has (2026-11-09: the generator works for more than one robot).

scenes/<scene>.yaml says it in a `robot:` block; a scene without one is the first robot (TOMO re_uhand) with its own USD, exactly as
before. Nothing about a robot lives in the generator's code outside this file.

    robot:
      object: tomo_gidora_ehand_Object     # the robot's root object in the .blend
      armatures: {}                        # joint-name prefix -> armature, for hands that are armatures of their own (default: none)
      axes: {}                             # joint -> bone axis 0/1/2 it turns about (default Z = 2)
      joints: [...]                        # the dataset's joint order (default: the URDF's revolute joints, in its order)
      urdf: ../path/robot.urdf             # relative to the scene file: limits, self-collision, the objects export
      srdf: ../path/robot.srdf
      drawn: scene                         # "scene" = draw the .blend's own robot; "usd" = replace it by the robot USD in renderer/
      limits_stride: 1                     # the self-collision check looks at every Nth frame (the check is a report, not a filter)
      cup_links: "{hand}_link_{k}endeff"   # the fingertips, k = 2..5: fit_dataset_to_limits.py keeps their centre in place
    camera:
      object: Camera.001                   # the head camera in the .blend (default: the robot USD's camera)
      lens: [fx, fy, cx, cy, W, H]         # the real camera's own numbers (default: the first robot's lens)

    env/bin/python scene_robot.py <scene> urdf|srdf|object      -> one line, for generate.sh
"""
from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FIRST_ROBOT_URDF = HERE / "robot_limits/source/robot_description.urdf"
FIRST_ROBOT_SRDF = HERE / "robot_limits/source/tomo_config.srdf"
# the first robot's finger bases: found against the stored motion (max 0.034 deg over all 1116 frames)
FIRST_ROBOT_AXES = {"RF_joint_21": 0, "RF_joint_31": 1, "RF_joint_41": 1, "RF_joint_51": 0,
                    "LF_joint_21": 1, "LF_joint_31": 1, "LF_joint_41": 1, "LF_joint_51": 1}


def urdf_limits(urdf, margin_deg):
    """The joints table of robot_limits/tomo_re_uhand_limits.yaml, from a URDF."""
    m = math.radians(margin_deg)
    out = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        lim = j.find("limit")
        if j.get("type") == "revolute" and lim is not None:
            lo, hi = float(lim.get("lower")), float(lim.get("upper"))
            out[j.get("name")] = dict(lower_rad=lo, upper_rad=hi, safe_lower_rad=lo + m, safe_upper_rad=hi - m,
                                      lower_deg=round(math.degrees(lo), 2), upper_deg=round(math.degrees(hi), 2))
    return out


def urdf_joints(urdf):
    """The revolute joints, in the file's order (the MMRT recorder's order too)."""
    return [j.get("name") for j in ET.parse(urdf).getroot().findall("joint") if j.get("type") == "revolute"]


def robot_setup(cfg, scene_dir=HERE / "scenes"):
    """cfg = scenes/<scene>.yaml -> {object, armatures, axes, joints, urdf, srdf, drawn} (urdf, srdf: absolute paths)."""
    r = cfg.get("robot")
    if not r:
        return {"object": "tomo", "armatures": {"RF": "tomo_right_hand", "LF": "tomo_left_hand"}, "axes": dict(FIRST_ROBOT_AXES),
                "joints": json.loads((HERE / "renderer/tools/joint_names.json").read_text()),
                "urdf": str(FIRST_ROBOT_URDF), "srdf": str(FIRST_ROBOT_SRDF), "drawn": "usd", "limits_stride": 1,
                "cup_links": "{hand}_link_{k}endeff"}
    path = lambda k: str((Path(scene_dir) / r[k]).resolve()) if r.get(k) else None  # noqa: E731
    urdf = path("urdf")
    joints = r.get("joints") or (urdf_joints(urdf) if urdf else None)
    if not joints:
        raise SystemExit("scene file: robot needs joints, or a urdf to read them from")
    return {"object": r["object"], "armatures": dict(r.get("armatures") or {}), "axes": dict(r.get("axes") or {}), "joints": list(joints),
            "urdf": urdf, "srdf": path("srdf"), "drawn": r.get("drawn", "scene"), "limits_stride": int(r.get("limits_stride", 1)),
            "cup_links": r.get("cup_links", "{hand}_link_{k}endeff")}


def recorder_steps(doc):
    """rig.json -> [[step, first frame], ...]: the MMRT recorder's sub-episodes stored in the .blend; None if it has none."""
    eps = sorted((e["start_frame"], e["episode_name"]) for e in (doc["scene_mmrt"].get("mmrt_recorder") or {}).get("episodes") or [])
    if not eps:
        return None
    return ([["START", 0]] if eps[0][0] > 0 else []) + [[n, f] for f, n in eps]


def scene_setup(scene):
    p = HERE / "scenes" / f"{scene}.yaml"
    return robot_setup(yaml.safe_load(p.read_text()) if p.exists() else {}, p.parent)


if __name__ == "__main__":
    print(scene_setup(sys.argv[1])[sys.argv[2]] or "")
