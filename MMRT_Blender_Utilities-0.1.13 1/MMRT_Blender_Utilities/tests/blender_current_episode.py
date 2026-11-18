"""Render a tiny actual dataset episode with retimed NLA and verify restoration."""
import ast
import os
from pathlib import Path
import sys
import tempfile
import traceback
import numpy as np
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import MMRT_Blender_Utilities as addon
from MMRT_Blender_Utilities.common.synthesis.current_episode import export_current_episode, _read_metadata
from MMRT_Blender_Utilities import joint_limit_retime as retime


def run():
    addon.register()
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 2
    scene.render.fps = 24
    scene.render.fps_base = 1.001
    scene.render.frame_map_new = 200
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.mmrt_recorder.config.env_export_mode = 'OFF'
    scene.mmrt_recorder.config.export_binary = True
    urdf = bpy.data.texts.new('test.urdf')
    urdf.write('<robot name="test"><link name="base"/>' + ''.join(
        f'<link name="l{i}"/><joint name="j{i}" type="revolute"><parent link="base"/><child link="l{i}"/>'
        '<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1"/></joint>' for i in range(3)) + '</robot>')
    srdf = bpy.data.texts.new('test.srdf')
    srdf.write('<robot name="test"/>')
    objects = []
    for i, name in enumerate(('tomo', 'tomo_left_hand', 'tomo_right_hand')):
        bpy.ops.object.armature_add()
        obj = bpy.context.object
        obj.name = name
        bone = obj.pose.bones[0]
        bone.rotation_mode = 'XYZ'
        for frame, value in ((1, 0), (2, .1)):
            bone.rotation_euler.x = value
            bone.keyframe_insert('rotation_euler', index=0, frame=frame)
        robot = scene.mmrt_recorder.robots.add()
        robot.obj, robot.urdf_text, robot.srdf_text = obj, urdf, srdf
        pair = robot.joints.add()
        pair.joint_name, pair.bone_name = f'j{i}', bone.name
        objects.append(obj)
    scene[retime.STATE] = retime._snapshot(scene, objects)
    for strip in retime._make_strips(scene):
        retime._set_clock(strip, np.array([1., 1.5, 2.]), np.array([1., 2., 3.]), np.array([.5, .5, .5]))
    scene.frame_end = 3
    scene.frame_step = 2
    scene.frame_set(2, subframe=.25)
    source_tree = scene.compositing_node_group
    source_path = scene.render.filepath
    scene.mmrt_synthesizer.episode_prefix = 'current'
    scene.mmrt_synthesizer.episode_start_index = 1
    scene.mmrt_synthesizer.dry_run = True  # This dedicated export ignores batch dry-run/randomization.
    scene.mmrt_synthesizer.save_blend = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scene.mmrt_synthesizer.output_root = tmp
        (root / 'metadata.py').write_text("metadata = {'episodes': {'existing_001': {'keep': True}}}\n")
        directory = export_current_episode(bpy.context)
        assert (directory/'current_001.mp4').stat().st_size > 0
        assert len(list((directory/'depth').glob('*.exr'))) == 5
        text = (directory/'current_001.mmrt.yaml').read_text()
        assert 'total_frames: 5' in text and 'start_frame: 2' in text and 'end_frame: 6' in text
        assert '23.976' in text
        for name in ('tomo','tomo_left_hand','tomo_right_hand'):
            assert name+'.location.x' in text
        assert (directory/'current_001.urdf').exists() and (directory/'current_001.srdf').exists()
        assert (directory/'current_001.blend').exists()
        meta = _read_metadata(root/'metadata.py')
        assert meta['episodes']['existing_001'] == {'keep': True}
        assert meta['episodes']['current_001']['regional_retiming_active']
        assert meta['episodes']['current_001']['offsets_applied'] == {}
        assert retime.STATE in scene and all(obj.animation_data.action is None for obj in objects)
        assert scene.frame_end == 3 and scene.frame_step == 2 and scene.frame_current == 2
        assert abs(scene.frame_subframe-.25) < 1e-5
        assert scene.render.filepath == source_path and scene.compositing_node_group == source_tree
        assert scene.mmrt_recorder.config.export_binary
        if os.environ.get('MMRT_TEST_OUTPUT'):
            import shutil
            shutil.copytree(directory, os.environ['MMRT_TEST_OUTPUT'])
        try:
            export_current_episode(bpy.context)
        except ValueError as exc:
            assert 'already exists' in str(exc)
        else:
            raise AssertionError('Existing episode overwritten')
        from MMRT_Blender_Utilities.common.synthesis.pipeline import BlenderSynthesizer
        original_record = BlenderSynthesizer.record_mmrt
        def fail_record(*args):
            raise RuntimeError('forced recording failure')
        scene.mmrt_synthesizer.episode_start_index = 2
        BlenderSynthesizer.record_mmrt = fail_record
        try:
            try:
                export_current_episode(bpy.context)
            except RuntimeError as exc:
                assert 'forced recording failure' in str(exc)
            else:
                raise AssertionError('Failure was ignored')
        finally:
            BlenderSynthesizer.record_mmrt = original_record
        assert 'current_002' not in _read_metadata(root/'metadata.py')['episodes']
        assert scene.compositing_node_group == source_tree and scene.frame_end == 3
        assert scene.mmrt_recorder.config.export_binary and retime.STATE in scene
        retime.restore_timing(scene)
        scene[retime.STATE] = retime._snapshot(scene, objects)
        scene[retime.STATE]['method'] = 'KEYFRAMES'
        retime.make_working_actions(scene[retime.STATE])
        retime.apply_key_clock(scene[retime.STATE], [1.,2.], [1.,3.], retime._curves)
        scene.frame_end = 3
        scene.mmrt_synthesizer.episode_start_index = 3
        scene.mmrt_synthesizer.save_blend = False
        direct_directory = export_current_episode(bpy.context)
        assert (direct_directory/'current_003.mp4').stat().st_size > 0
        assert 'total_frames: 5' in (direct_directory/'current_003.mmrt.yaml').read_text()
        assert all(obj.animation_data.action and not obj.animation_data.nla_tracks for obj in objects)
        assert scene[retime.STATE]['method'] == 'KEYFRAMES'
        print('PASS direct existing-key animation exports dataset video/MMRT without NLA',flush=True)
        print('PASS actual MP4, five depth EXRs, MMRT, robot bundle, merged metadata, NLA/state preservation and overwrite refusal',flush=True)
    addon.unregister()

try:
    run()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
