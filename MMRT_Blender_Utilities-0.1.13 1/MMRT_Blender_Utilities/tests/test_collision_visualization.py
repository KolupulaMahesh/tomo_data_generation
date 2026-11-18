"""Regression coverage for atomic-group collision ownership and plotting."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "processes" / "mmrt"))
from common.visualization import Visualization


@pytest.mark.parametrize("contacts, expected", [
    ([(0, 7, 99, 0, 1)], {"arm"}),  # environment link 1 is not the gripper
    ([(0, 99, 7, 0, 1)], {"gripper"}),  # robot on side B
    ([(0, 7, 7, 0, 1)], {"arm", "gripper"}),
    ([(0, 7, 99, 2, -1)], {"gripper"}),  # fixed tool inherits gripper
    ([(0, 7, 99, -1, 0)], set()),  # unowned base
    ([], set()),
])
def test_contact_ownership(monkeypatch, contacts, expected):
    pytest.importorskip("pybullet")
    from common import simulation

    def joint_info(robot, index):
        info = [None] * 17
        info[1] = [b"arm_joint", b"gripper_joint", b"fixed_tool"][index]
        info[16] = [-1, 0, 1][index]
        return info

    monkeypatch.setattr(simulation.p, "getNumJoints", lambda robot: 3)
    monkeypatch.setattr(simulation.p, "getJointInfo", joint_info)
    sim = simulation.Simulation.__new__(simulation.Simulation)
    assert sim.get_atomic_groups_for_contacts(
        7, contacts, {"arm": ["arm_joint"], "gripper": ["gripper_joint"]}
    ) == expected


def test_red_lines_only_on_contacted_groups(tmp_path):
    data = SimpleNamespace(frame_start=100, flags_general={}, flags_io={})
    vis = Visualization(
        {"arm": [0, 1, 1, 0], "gripper": [0, 1, 1, 0], "tool": [0] * 4},
        {"arm": [(0, 3)], "gripper": [(0, 3)], "tool": []},
        binned_segments=[(0, 3, "combined", [])],
        mmrt_data=data,
        collision_frames=[101, 102],
        collision_frames_by_atomic_group={"arm": [101, 101], "tool": [102]},
    )
    fig = vis.plot(save_path=tmp_path / "plot.svg", show=False)
    try:
        red_lines = [[line for line in ax.lines if line.get_color() == "red"]
                     for ax in fig.axes]
        assert [len(lines) for lines in red_lines] == [1, 0, 1, 0]
        assert list(red_lines[0][0].get_xdata()) == [1, 1]
        assert list(red_lines[2][0].get_xdata()) == [2, 2]
        assert red_lines[0][0].get_label() == "Collision"
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_empty_ownership_does_not_fall_back_to_overlapping_segments():
    vis = Visualization({"arm": [1]}, {"arm": [(0, 1)]},
                        collision_frames=[0], collision_frames_by_atomic_group={})
    assert vis._collision_frames_for_group("arm") == []
