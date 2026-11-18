"""How far each hand moved between two MMRT recordings (e.g. original vs fixed): position and turn, per frame."""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))          # kinematics/ = a frozen copy of the brain's FK
from kinematics.fk import RobotFK                                # noqa: E402

fk = RobotFK(str(Path(__file__).resolve().parent / "source/robot_description.urdf"))


def load(p):
    d = yaml.safe_load(open(p))
    return [x for v in d["labels"].values() for x in v], np.asarray(d["frames"], float)


la, A = load(sys.argv[1])
lb, B = load(sys.argv[2])
for side in ("Left", "Right"):
    dpos, drot = [], []
    for f in range(len(A)):
        Ta = fk.link_transform(f"{side}_hand", dict(zip(la, A[f])))
        Tb = fk.link_transform(f"{side}_hand", dict(zip(lb, B[f])))
        dpos.append(1000 * np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
        drot.append(np.degrees(np.arccos(np.clip((np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1) / 2, -1, 1))))
    dpos, drot = np.array(dpos), np.array(drot)
    big = np.flatnonzero(dpos > 5)
    print(f"  {side} hand: moved up to {dpos.max():6.1f} mm (frame {dpos.argmax()}), turned up to {drot.max():5.1f} deg; "
          f"frames moved > 5 mm: {len(big)}" + (f" ({big[0]}-{big[-1]})" if len(big) else ""))
