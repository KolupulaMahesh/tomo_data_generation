import importlib.util
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location('key_clock', Path(__file__).resolve().parents[1]/'common/keyframe_retime.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_overlapping_segments_share_one_affine_stretch():
    source, target = core.shared_key_clock([0, 5, 10, 15, 20], [0, 5, 20, 25, 30], [(0, 10), (5, 20)])
    assert np.allclose(source, [0,20])
    assert np.allclose(target, [0,60])


def test_separated_movements_preserve_gap_duration():
    source, target = core.shared_key_clock([0, 10, 20, 30], [0, 20, 30, 60], [(0,10),(20,30)])
    assert np.allclose(source,[0,10,20,30])
    assert np.allclose(target,[0,20,30,60])
    assert core.map_frame(15,source,target)==25
    assert core.map_frame(35,source,target)==65
    assert core.map_frame(-5,source,target)==-5


def test_no_motion_keeps_original_timing():
    source, target = core.shared_key_clock([0,10,20],[0,20,40],[])
    assert np.allclose(source,target)


def test_speedup_allows_other_regions_but_rejects_boundary_violations():
    source, target = [0,10,20,30], [0,5,15,35]
    def issue(frame, metric='velocity'):
        return dict(source_frame=frame, metric=metric)
    outside = issue(18)
    inside = issue(6, 'acceleration')
    boundary = issue(12, 'acceleration')
    regions, failures = core.speedup_violations([outside, inside, boundary], source, target)
    assert regions == [(-2.,7.)]
    assert failures == [inside, boundary]
    assert core.speedup_violations([outside], source, target)[1] == []
    assert core.speedup_violations([issue(18, 'position')], source, target)[1]


def test_speedup_boundary_guard_uses_output_frames():
    issue = dict(source_frame=11.5, metric='acceleration')
    assert core.speedup_violations([issue], [0,10,20], [0,5,15], 1)[1]
    assert not core.speedup_violations([issue], [0,10,20], [0,5,15], 2)[1]
    assert not core.speedup_violations([issue], [0,10,20], [0,10,20])[0]
